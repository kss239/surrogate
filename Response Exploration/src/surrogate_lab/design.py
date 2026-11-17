"""Diverse feasible new configurations and local replication proposals; no simulator execution."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata
from sklearn.ensemble import RandomForestRegressor
from .data import summarize_runs, scale_bounds
from .validation import support_distance
from .uncertainty import predict_uncertainty


def unit_rank(values):
    values = np.asarray(values)
    return (rankdata(values, method="average")-1)/max(1, len(values)-1)


def propose_batch(state, data, oof, cfg, feasibility, reserved_X):
    b = np.asarray(cfg.bounds)
    Z = qmc.Sobol(len(cfg.factors), scramble=True, seed=cfg.seed+900).random_base2(int(np.ceil(np.log2(cfg.candidate_count))))[:cfg.candidate_count]
    X = b[:,0]+Z*(b[:,1]-b[:,0])
    observed = summarize_runs(data, cfg)
    excluded = np.vstack([observed[list(cfg.factors)].to_numpy(), reserved_X])
    mask = np.asarray(feasibility(X), bool) & (support_distance(X, excluded, cfg)>1e-10)
    X = X[mask]
    if len(X) < cfg.new_alternatives: raise ValueError("Too few feasible new candidates")
    p = predict_uncertainty(state, X, cfg)
    error_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=cfg.seed, n_jobs=1)
    error_model.fit(scale_bounds(oof[list(cfg.factors)], cfg), abs(oof.residual))
    expected_abs_error = error_model.predict(scale_bounds(X, cfg))
    score = .4*unit_rank(p.bootstrap_mean_sd_approx)+.35*unit_rank(p.support_distance)+.25*unit_rank(expected_abs_error)
    chosen=[]
    for _ in range(cfg.new_alternatives):
        diversity = np.ones(len(X)) if not chosen else np.minimum(1., support_distance(X, X[chosen], cfg)/.7)
        adjusted = score*(.15+.85*diversity)
        adjusted[chosen] = -np.inf
        chosen.append(int(np.argmax(adjusted)))
    new = pd.DataFrame(X[chosen], columns=cfg.factors).assign(
        action="new_alternative", group=np.nan, repetitions=cfg.new_repetitions,
        acquisition_score=score[chosen], expected_local_mean_variance_reduction=np.nan,
        reason="coverage + bootstrap disagreement + predicted development error")
    # With no within-alternative estimate, prioritize obtaining replication;
    # do not present an invented numerical variance reduction.
    reduction = observed.variance*(1/observed.reps - 1/(observed.reps+cfg.additional_repetitions))
    repeat = observed.assign(expected_local_mean_variance_reduction=reduction,
        needs_replication=observed.reps < 2).sort_values(
            ['needs_replication', 'expected_local_mean_variance_reduction'], ascending=[False,False]).head(cfg.repeat_alternatives)
    repeat = repeat[[*cfg.factors, "group", "expected_local_mean_variance_reduction"]].assign(
        action="additional_repetitions", repetitions=cfg.additional_repetitions,
        acquisition_score=np.nan, reason="largest estimated local sampling-variance reduction")
    repeat.loc[repeat.expected_local_mean_variance_reduction.isna(), 'reason'] = "obtain replication: within-alternative noise unknown"
    return pd.concat([new, repeat], ignore_index=True).assign(estimated_cost=lambda d: d.repetitions*cfg.cost_per_run)
