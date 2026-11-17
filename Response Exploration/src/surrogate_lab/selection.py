"""Development-only finalist comparison, ensemble construction, and seed checks."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from .search import leaderboard, restore_recipe
from .validation import evaluate_recipe, metrics


def shortlist_records(records, n=3):
    ids = leaderboard(records).drop_duplicates("family").head(n).trial_id.tolist()
    lookup = {r["trial_id"]: r for r in records}
    return tuple(lookup[i] for i in ids)


def ensemble_predictions(frames):
    keys = ["split", "group"]
    base = frames[0].set_index(keys).copy()
    combined = np.column_stack([f.set_index(keys).loc[base.index, "predicted"].to_numpy() for f in frames])
    return base.assign(predicted=combined.mean(axis=1), residual=base["mean"]-combined.mean(axis=1)).reset_index()


def evaluate_finalists(shortlist, data, challenges, cfg):
    regional = tuple(evaluate_recipe(restore_recipe(r), data, challenges, cfg) for r in shortlist)
    summaries = []
    for r, stress in zip(shortlist, regional):
        worst = max(s["rmse"] for s in stress["fold_scores"])
        summaries.append({"candidate": r["trial_id"], "cv_rmse": r["metrics"]["rmse"], "worst_region_rmse": worst,
                          "selection_score": r["metrics"]["rmse"]+cfg.regional_penalty*worst})
    cv_ensemble = ensemble_predictions([pd.DataFrame(r["predictions"]) for r in shortlist])
    stress_ensemble = ensemble_predictions([pd.DataFrame(r["predictions"]) for r in regional])
    rmse = metrics(cv_ensemble["mean"], cv_ensemble.predicted, cfg.tolerance)["rmse"]
    worst = max(metrics(g["mean"], g.predicted, cfg.tolerance)["rmse"] for _,g in stress_ensemble.groupby("split"))
    summaries.append({"candidate": "equal_ensemble", "cv_rmse": rmse, "worst_region_rmse": worst, "selection_score": rmse+cfg.regional_penalty*worst})
    ranking = pd.DataFrame(summaries).sort_values("selection_score")
    selected_id = ranking.iloc[0].candidate
    selected = tuple(restore_recipe(r) for r in shortlist if selected_id == "equal_ensemble" or r["trial_id"] == selected_id)
    oof = cv_ensemble if selected_id == "equal_ensemble" else pd.DataFrame(next(r for r in shortlist if r["trial_id"] == selected_id)["predictions"])
    return {"ranking": ranking, "regional": regional, "recipes": selected, "oof": oof}

def seed_stability(recipes, data, splits, cfg, seeds=(2026, 2027, 2028)):
    """Fixed recipes across seeds: development diagnostic, not final-test evidence."""
    rows = []
    for seed in seeds:
        fitted = [evaluate_recipe(replace(r, seed=seed), data, splits, cfg) for r in recipes]
        oof = ensemble_predictions([pd.DataFrame(r["predictions"]) for r in fitted])
        rows.append({"seed": seed, **metrics(oof["mean"], oof.predicted, cfg.tolerance)})
    return pd.DataFrame(rows)
