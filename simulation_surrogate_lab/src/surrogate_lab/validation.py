"""Shared validation splits, metrics, and held-out evaluation; no tuning or I/O."""
from __future__ import annotations
from dataclasses import asdict
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data import summarize_runs, subset, scale_bounds
from .models import fit_recipe, predict_fit


def metrics(y, pred, tolerance):
    y, pred = np.asarray(y), np.asarray(pred)
    error = y-pred
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": float(mean_absolute_error(y, pred)),
            "bias_observed_minus_predicted": float(error.mean()), "p90_abs_error": float(np.quantile(abs(error), .9)),
            "within_tolerance": float(np.mean(abs(error)<=tolerance)), "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan")}


def support_distance(X, reference, cfg):
    return NearestNeighbors(n_neighbors=1).fit(scale_bounds(reference, cfg)).kneighbors(scale_bounds(X, cfg))[0][:, 0]


def distributed_splits(data, cfg, folds=None):
    groups = np.sort(data.group.unique())
    support=np.array([],dtype=groups.dtype)
    if cfg.split_strategy == 'boundary':
        if 'boundary_support' in data:
            support=np.sort(data.loc[data.boundary_support,'group'].unique())
        else:
            from .boundary import boundary_support
            support=boundary_support(data,cfg)['selected'].group.to_numpy()
        groups=groups[~np.isin(groups,support)]
    count = min(folds or cfg.folds, len(groups))
    if count < 2 or len(support)+len(groups)-int(np.ceil(len(groups)/count)) < 2:
        raise ValueError("Grouped CV requires at least two training alternatives per fold; use more groups or fewer holdouts")
    return tuple((f"fold_{i}", np.concatenate([groups[tr],support]), groups[va]) for i, (tr, va) in enumerate(
        KFold(n_splits=count, shuffle=True, random_state=cfg.seed).split(groups)))


def challenge_splits(data, cfg):
    s = summarize_runs(data, cfg)
    X, g = s[list(cfg.factors)].to_numpy(), s.group.to_numpy()
    if len(s) < 3: raise ValueError("Regional challenges require at least three alternatives")
    labels = KMeans(n_clusters=min(3, len(s)), n_init=10, random_state=cfg.seed).fit_predict(scale_bounds(X, cfg))
    regions = [(f"cluster_{k}", g[labels != k], g[labels == k]) for k in np.unique(labels)
               if np.sum(labels != k) >= 2]
    # All factor boundaries in broad mode; first two for the executable smoke demonstration.
    for j in range(len(cfg.factors) if cfg.profile == "broad" else min(2, len(cfg.factors))):
        order = np.argsort(X[:, j], kind="stable")
        n = min(len(s)-2, max(1, int(.15*len(s))))
        for side, indices in (("low", order[:n]), ("high", order[-n:])):
            mask = np.isin(np.arange(len(s)), indices)
            regions.append((f"{cfg.factors[j]}_{side}", g[~mask], g[mask]))
    return tuple(regions)


def evaluate_recipe(recipe, data, splits, cfg):
    started = time.perf_counter()
    predictions, scores, notices, run_predictions = [], [], [], []
    for name, tr, va in splits:
        if set(tr) & set(va): raise ValueError("Overlapping train/validation groups")
        train, valid = subset(data, tr), summarize_runs(subset(data, va), cfg)
        fitted = fit_recipe(train, recipe, cfg)
        X = valid[list(cfg.factors)].to_numpy()
        pred = predict_fit(fitted, X, cfg)
        if not np.isfinite(pred).all(): raise ValueError("Nonfinite predictions")
        distance = support_distance(X, summarize_runs(train, cfg)[list(cfg.factors)].to_numpy(), cfg)
        scored = metrics(valid["mean"], pred, cfg.tolerance)
        scores.append({"split": name, **scored})
        joined = subset(data, va).merge(valid[['group']].assign(predicted=pred), on='group', validate='many_to_one')
        run_predictions.append(joined)
        predictions.extend(valid.assign(predicted=pred, residual=valid["mean"]-pred,
                                          distance=distance, split=name).to_dict("records"))
        notices.extend(fitted["warnings"])
    p = pd.DataFrame(predictions)
    runs = pd.concat(run_predictions, ignore_index=True)
    return {"status": "complete", "recipe": asdict(recipe), "seconds": time.perf_counter()-started,
            "metrics": metrics(p["mean"], p.predicted, cfg.tolerance),
            "run_metrics": metrics(runs[cfg.response], runs.predicted, cfg.tolerance),
            "fold_rmse_sd": float(np.std([s["rmse"] for s in scores])),
            "fold_scores": scores, "predictions": predictions, "warnings": sorted(set(notices))}
