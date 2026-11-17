"""Post-fit evidence and hypotheses; never modifies the fitted predictor."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import summarize_runs
from .models import predict_bundle
from .features import transform_features
from .diagnostics import coverage_report, residual_patterns, tail_diagnostics, noise_summary


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Descriptive screening cutoffs, not hypothesis-test significance levels."""
    flat_ratio: float = .10
    compressed_ratio: float = .50
    minimum_skill: float = .05
    large_skill_gap: float = .25


def prediction_statistics(observed, predicted, baseline, tolerance):
    y, p = np.asarray(observed, float), np.asarray(predicted, float)
    if y.ndim != 1 or p.shape != y.shape or len(y) < 1:
        raise ValueError("Need aligned one-dimensional arrays with at least one observation")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(baseline):
        raise ValueError("Nonfinite observation, prediction, or baseline")
    sy, sp = float(y.std()), float(p.std())
    eps = 1e-12 * max(1., float(np.max(abs(y))), abs(baseline))
    mse = float(np.mean((y-p)**2))
    base_mse = float(np.mean((y-baseline)**2))
    ratio = sp/sy if sy > eps else np.nan
    corr = float(np.corrcoef(y,p)[0,1]) if min(sy,sp)>eps else np.nan
    slope = float(np.mean((y-y.mean())*(p-p.mean()))/sy**2) if sy>eps else np.nan
    return {"n":len(y),"observed_sd":sy,"predicted_sd":sp,"sd_ratio":ratio,
            "predicted_on_observed_slope":slope,"correlation":corr,
            "rmse":np.sqrt(mse),"mae":float(np.mean(abs(y-p))),
            "r2":1-mse/sy**2 if sy>eps else np.nan,
            "training_mean_baseline_rmse":np.sqrt(base_mse),
            "skill_vs_training_mean":1-mse/base_mse if base_mse>eps**2 else np.nan,
            "prediction_minus_observation_bias":float(np.mean(p-y)),
            "predicted_mean_minus_training_mean":float(p.mean()-baseline),
            "within_tolerance":float(np.mean(abs(p-y)<=tolerance))}


def inspect_inputs(bundle, train_summary, held_summary, cfg):
    """Inspect fitted feature transforms; never refit them using evaluation data."""
    Xt=train_summary[list(cfg.factors)].to_numpy(float)
    Xh=held_summary[list(cfg.factors)].to_numpy(float)
    rows=[]
    for i,fitted in enumerate(bundle):
        t=fitted['scaler'].transform(transform_features(Xt,fitted['features'],cfg))
        h=fitted['scaler'].transform(transform_features(Xh,fitted['features'],cfg))
        rows.append({"member":i,"family":fitted['recipe'].family,"feature_count":t.shape[1],
            "constant_training_features":int(np.sum(np.ptp(t,axis=0)<=1e-12)),
            "constant_heldout_features":int(np.sum(np.ptp(h,axis=0)<=1e-12)),
            "finite_features":bool(np.isfinite(t).all() and np.isfinite(h).all()),
            "heldout_values_outside_training_range_fraction":float(np.mean((h<t.min(axis=0))|(h>t.max(axis=0))))})
    return pd.DataFrame(rows)


def postfit_report(bundle, train_runs, heldout_runs, cfg, thresholds=None, observed_predictions=None):
    """Compare in-sample training and held-out behavior at both observation grains.

    observed_predictions, if supplied, must have one row per group with all original
    factors and a predicted_mean column. Keys and factor values are checked before joining.
    Numerical checks cannot establish upstream simulator-label correctness.
    Independent repetitions are assumed for the noise decomposition.
    """
    thresholds=thresholds or DiagnosticThresholds()
    train=summarize_runs(train_runs,cfg)
    held=summarize_runs(heldout_runs,cfg)
    if set(train.group)&set(held.group):
        raise ValueError("Training and held-out groups overlap")
    baseline=float(train['mean'].mean())
    tp=predict_bundle(bundle,train[list(cfg.factors)].to_numpy(),cfg)
    if observed_predictions is None:
        hp=predict_bundle(bundle,held[list(cfg.factors)].to_numpy(),cfg)
    else:
        required=['group',*cfg.factors,'predicted_mean']
        if not set(required)<=set(observed_predictions): raise ValueError("Prediction table missing keys/factors/predicted_mean")
        if observed_predictions.group.duplicated().any(): raise ValueError("Duplicate prediction keys")
        if set(observed_predictions.group)!=set(held.group): raise ValueError("Prediction keys do not match evaluation groups")
        aligned=observed_predictions.set_index('group').loc[held.group]
        if not np.allclose(aligned[list(cfg.factors)].to_numpy(float),held[list(cfg.factors)].to_numpy(float),rtol=0,atol=1e-12):
            raise ValueError("Factors disagree across prediction/observation join")
        hp=aligned.predicted_mean.to_numpy(float)
    frames={"training":train.assign(predicted=tp),"heldout":held.assign(predicted=hp)}
    rows=[]; run_frames={}
    for label,summary,data in [('training',frames['training'],train_runs),('heldout',frames['heldout'],heldout_runs)]:
        rows.append({"partition":label,"grain":"alternative_mean",**prediction_statistics(summary['mean'],summary.predicted,baseline,cfg.tolerance)})
        joined=data.merge(summary[['group','predicted']],on='group',validate='many_to_one')
        run_frames[label]=joined
        rows.append({"partition":label,"grain":"individual_run",**prediction_statistics(joined[cfg.response],joined.predicted,baseline,cfg.tolerance)})
    scores=pd.DataFrame(rows)
    t=scores.query("partition=='training' and grain=='alternative_mean'").iloc[0]
    h=scores.query("partition=='heldout' and grain=='alternative_mean'").iloc[0]
    # Equal-alternative summaries, unlike the row-weighted individual-run plot/metrics.
    noise_values=noise_summary(held)
    between=noise_values['between_mean_variance']
    reference_noise=noise_values['mean_sampling_variance']
    signal_raw=noise_values['signal_variance_raw']
    noise=pd.DataFrame([noise_values])
    inputs=inspect_inputs(bundle,train,held,cfg)
    coverage=coverage_report(train,held,cfg)
    held_details=frames['heldout'].merge(coverage['heldout_distances'], on='group', validate='one_to_one')
    residuals=residual_patterns(held_details,cfg)
    tails=tail_diagnostics(held_details,train['mean'],cfg)
    findings=[]
    def add(flag,evidence,check): findings.append({"finding":flag,"evidence":evidence,"suggested_check":check})
    if not np.isfinite(h.sd_ratio):
        add("Observed means essentially constant","Observed SD is near zero; spread ratio and R² are undefined","Use absolute errors and response tolerance; do not diagnose collapse from a ratio")
    elif h.sd_ratio<thresholds.flat_ratio:
        add("Near-flat held-out predictions",f"Prediction SD / observed-mean SD = {h.sd_ratio:.3f}","Inspect data alignment and transformed inputs; compare training behavior")
    elif h.sd_ratio<thresholds.compressed_ratio:
        add("Compressed held-out predictions",f"Prediction SD / observed-mean SD = {h.sd_ratio:.3f}","Check regularization, missing interactions, support and reference-mean noise")
    if np.isfinite(h.skill_vs_training_mean) and h.skill_vs_training_mean<thresholds.minimum_skill:
        add("Little improvement over mean-only baseline",f"MSE skill versus training mean = {h.skill_vs_training_mean:.3f}","Validate joins/response column before expanding model search")
    if np.isfinite(t.sd_ratio) and np.isfinite(h.sd_ratio) and max(t.sd_ratio,h.sd_ratio)<thresholds.flat_ratio:
        add("Both training and held-out predictions are flat","Small prediction spread at both partitions","Investigate weak factor signal, excessive constraints, constant features or upstream labeling")
    if np.isfinite(t.skill_vs_training_mean-h.skill_vs_training_mean) and t.skill_vs_training_mean-h.skill_vs_training_mean>thresholds.large_skill_gap:
        add("Training-to-heldout performance gap",f"MSE skill gap = {t.skill_vs_training_mean-h.skill_vs_training_mean:.3f}","Investigate overfitting, coverage gaps and preprocessing consistency; training score is optimistic")
    if (np.isfinite(t.skill_vs_training_mean) and t.skill_vs_training_mean < thresholds.minimum_skill
            and signal_raw > reference_noise):
        add("Possible underfitting despite mean signal", "Low training skill with estimated between-alternative signal above mean noise",
            "Compare regularized quadratic, additive and kernel models using development folds; this does not establish a cause")
    if (coverage['marginal'].occupied_bin_fraction < 1).any():
        add("Incomplete marginal DOE coverage", "Some physical-bound bins contain no training alternatives",
            "Inspect pairwise occupancy and held-out support distances; constrained domains can have intentionally empty bins")
    if abs(h.prediction_minus_observation_bias)>cfg.tolerance:
        add("Bias exceeds response tolerance",f"Prediction minus observation bias = {h.prediction_minus_observation_bias:.3f}","Inspect regional residuals and train/test mean shift")
    if reference_noise>=.5*between:
        add("Noisy reference means",f"Estimated mean sampling variance {reference_noise:.3g}; between-mean variance {between:.3g}","Consider more independent repetitions at reference alternatives; decomposition is uncertain")
    if (inputs.constant_training_features>0).any():
        add("Constant transformed training columns",str(inputs.constant_training_features.tolist()),"Inspect whether expected factors survived preprocessing; some constant basis columns can be harmless")
    if noise_values['singleton_alternatives']:
        add("Some alternative variances are unknown", f"{noise_values['singleton_alternatives']} held-out configurations have one run",
            "Means remain scoreable; collect repetitions to separate stochastic noise from response signal")
    if not findings: add("No configured red flag triggered","Descriptive thresholds passed","Review plots, practical tolerances and regional errors; passing flags is not model acceptance")
    return {"coverage":coverage,"residual_patterns":residuals,"tails":tails,"scores":scores,"findings":pd.DataFrame(findings),"noise":noise,"inputs":inputs,
        "means":frames,"runs":run_frames,"baseline":baseline,"thresholds":thresholds}
