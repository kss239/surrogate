"""Approximate mean sensitivity and calibrated one-future-run prediction intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .data import summarize_runs, scale_bounds
from .models import fit_bundle, predict_bundle
from .validation import support_distance, metrics


def bootstrap_runs(data, cfg, seed):
    rng = np.random.default_rng(seed)
    groups = np.sort(data.group.unique())
    draws = rng.choice(groups, size=len(groups), replace=True)
    pieces = []
    for new_group, old_group in enumerate(draws):
        block = data.loc[data.group == old_group]
        sampled = block.iloc[rng.integers(0, len(block), size=len(block))].copy()
        pieces.append(sampled.assign(group=new_group))
    return pd.concat(pieces, ignore_index=True)


def fit_uncertainty(data, recipes, cfg):
    bundle = fit_bundle(data, recipes, cfg)
    members = tuple(fit_bundle(bootstrap_runs(data, cfg, cfg.seed+500+b), recipes, cfg)
                    for b in range(cfg.bootstrap_members))
    s = summarize_runs(data, cfg)
    replicated = s.loc[s.reps >= 2]
    variance_model = None
    floor = np.nan
    if len(replicated):
        floor = max(float(replicated.variance.median())*.01, 1e-8)
        variance_model = RandomForestRegressor(n_estimators=160, min_samples_leaf=5, random_state=cfg.seed, n_jobs=1)
        variance_model.fit(scale_bounds(replicated[list(cfg.factors)], cfg), replicated.variance.to_numpy())
    return {"bundle": bundle, "members": members, "variance_model": variance_model, "variance_floor": floor,
            "noise_status": "replicated_subset_model" if len(replicated) else "unavailable_no_replication",
            "replicated_alternatives": len(replicated),
            "reference_X": s[list(cfg.factors)].to_numpy()}


def predict_uncertainty(state, X, cfg):
    pred = predict_bundle(state["bundle"], X, cfg)
    member_predictions = np.asarray([predict_bundle(b, X, cfg) for b in state["members"]])
    bootstrap_sd = member_predictions.std(axis=0, ddof=1)
    if state["variance_model"] is None:
        variance = np.full(len(X), np.nan)
        # Ordinary absolute-residual conformal calibration: no claim to a
        # separately identified simulation-noise component without repeats.
        run_scale = np.ones(len(X))
    else:
        variance = np.maximum(state["variance_model"].predict(scale_bounds(X, cfg)), state["variance_floor"])
        run_scale = np.sqrt(variance+bootstrap_sd**2)
    return pd.DataFrame({"predicted_mean": pred, "bootstrap_mean_sd_approx": bootstrap_sd,
                         "estimated_run_sd": np.sqrt(variance),
                         "run_scale": run_scale,
                         "noise_status": state["noise_status"],
                         "support_distance": support_distance(X, state["reference_X"], cfg)})


def one_run_per_group(data, cfg, seed):
    return pd.concat([g.sample(n=1, random_state=seed+int(i)) for i,g in data.groupby("group")], ignore_index=True)


def conformal_quantile(scores, alpha):
    values = np.sort(np.asarray(scores, float))
    rank = int(np.ceil((len(values)+1)*(1-alpha)))
    return float(values[rank-1]) if rank <= len(values) else float("inf")


def calibrate_runs(state, data, cfg):
    reference = one_run_per_group(data, cfg, cfg.seed+800)
    pred = predict_uncertainty(state, reference[list(cfg.factors)].to_numpy(), cfg)
    scores = abs(reference[cfg.response].to_numpy()-pred.predicted_mean.to_numpy())/pred.run_scale.to_numpy()
    return {"scores": scores, "groups": reference.group.to_numpy(), "alpha": cfg.alpha,
            "q": conformal_quantile(scores, cfg.alpha), "target": "one future run at a new alternative"}


def predict_new(state, calibration_state, X, cfg, feasibility):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != len(cfg.factors) or not np.isfinite(X).all():
        raise ValueError("Expected finite matrix in configured factor order")
    b = np.asarray(cfg.bounds)
    valid = np.all((X >= b[:,0]) & (X <= b[:,1]), axis=1) & np.asarray(feasibility(X), bool)
    if not valid.all(): raise ValueError("Prediction request contains infeasible/out-of-bounds points")
    p = predict_uncertainty(state, X, cfg)
    width = calibration_state["q"]*p.run_scale
    return p.assign(run_interval_lower=p.predicted_mean-width, run_interval_upper=p.predicted_mean+width,
                    nominal_run_coverage=1-cfg.alpha)


def final_evaluation(state, calibration_state, test_data, cfg, feasibility):
    s = summarize_runs(test_data, cfg)
    pred = predict_new(state, calibration_state, s[list(cfg.factors)].to_numpy(), cfg, feasibility)
    means = pd.concat([s.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    reference = one_run_per_group(test_data, cfg, cfg.seed+1000).set_index("group").loc[s.group]
    means = means.assign(test_run=reference[cfg.response].to_numpy())
    means = means.assign(covered=(means.test_run>=means.run_interval_lower)&(means.test_run<=means.run_interval_upper),
                         interval_width=means.run_interval_upper-means.run_interval_lower)
    mean_scores = metrics(means["mean"], means.predicted_mean, cfg.tolerance)
    coverage=[]
    for level in (.5, .8, .9, .95):
        q = conformal_quantile(calibration_state["scores"], 1-level)
        covered = abs(means.test_run-means.predicted_mean)<=q*means.run_scale
        coverage.append({"nominal": level, "observed": float(covered.mean()), "mean_width": float((2*q*means.run_scale).mean())})
    run_predictions = test_data.merge(means[['group', 'predicted_mean']], on='group', validate='many_to_one')
    bands = pd.qcut(means.support_distance, 3, duplicates='drop')
    regional = means.groupby(bands, observed=True).agg(alternatives=('group','size'),
        coverage=('covered','mean'), mean_width=('interval_width','mean')).reset_index()
    return {"regional_coverage": regional,
            "run_metrics": metrics(run_predictions[cfg.response], run_predictions.predicted_mean, cfg.tolerance),
            "means": means, "mean_metrics": mean_scores, "coverage_curve": pd.DataFrame(coverage),
            "run_coverage": float(means.covered.mean()), "run_interval_width": float(means.interval_width.mean())}
