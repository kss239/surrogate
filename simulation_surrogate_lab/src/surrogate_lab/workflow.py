"""Explicit orchestration; development and final evaluation remain separate calls."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .data import audit_runs, study_inventory, summarize_runs, split_groups, subset
from .validation import distributed_splits, challenge_splits, metrics
from .search import optional_model_status, make_recipes, run_search, refinement_recipes, leaderboard
from .selection import shortlist_records, evaluate_finalists
from .storage import package_fingerprint, data_hash
from .models import fit_bundle, predict_bundle


def box_feasible(X):
    """Default additional constraint: all points allowed; physical bounds checked separately."""
    return np.ones(len(X), dtype=bool)


def prepare_study(raw, cfg, feasibility=box_feasible, manifest=None):
    """Audit and partition. Reuse a saved manifest only for exactly the same group assignment.

    To add runs safely, persist semantic configuration keys and explicitly extend partitions;
    integer group IDs are generated from sorted factor vectors and can change with new alternatives.
    """
    runs = audit_runs(raw, cfg, feasibility)
    summary = summarize_runs(runs, cfg)
    if manifest is not None:
        if manifest.get("data_hash") != data_hash(runs):
            raise ValueError("Saved manifest belongs to a different audited dataset; extend partitions explicitly")
        parts = {k: np.asarray(v) for k,v in manifest["partitions"].items()}
    else:
        parts = split_groups(summary,cfg)
    if set(parts) != {"development", "calibration", "test"}:
        raise ValueError("Manifest must contain development, calibration, test")
    groups = [set(x) for x in parts.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Partition overlap")
    if set.union(*groups) != set(runs.group):
        raise ValueError("Manifest does not match dataset groups")
    from .boundary import boundary_support
    boundary = boundary_support(runs,cfg) if cfg.split_strategy == 'boundary' else None
    if boundary is not None:
        support=set(boundary['selected'].group)
        if not support <= set(parts['development']): raise ValueError("Manifest does not retain boundary support in development")
        runs=runs.assign(boundary_support=runs.group.isin(support))
        boundary['partition_sizes']={k:len(v) for k,v in parts.items()}
    return {"boundary":boundary,"inventory": study_inventory(runs,cfg), "runs": runs, "summary": summary, "partitions": parts,
            **{k: subset(runs,v) for k,v in parts.items()}}


def develop(data, cfg, recipes=None):
    """Search and select using development data only; pass no held-out outcomes here."""
    splits = distributed_splits(data,cfg)
    planned = make_recipes(cfg,data.group.nunique()) if recipes is None else tuple(recipes)
    fingerprint = package_fingerprint()
    initial = run_search(planned,data,splits,cfg,fingerprint)
    records = initial["records"]
    refinement = None
    if cfg.profile == "broad":
        refinement = run_search(refinement_recipes(records,cfg),data,splits,cfg,fingerprint)
        records = tuple({r["trial_id"]:r for r in (*records,*refinement["records"])}.values())
    board = leaderboard(records)
    shortlist = shortlist_records(records,2 if cfg.profile == "smoke" else 4)
    finalists = evaluate_finalists(shortlist,data,challenge_splits(data,cfg),cfg)
    return {"optional_models":optional_model_status(cfg),"initial":initial,"refinement":refinement,"records":records,"board":board,
            "finalists":finalists,"splits":splits,"source_hash":fingerprint}


def nested_evaluate(data, cfg, outer_folds=3, recipes=None):
    """Optional full selection evaluation; reruns development search inside each outer fold.

    Uses only the supplied data (normally development). Expensive; no uncertainty calibration.
    If recipes is given, the scope is selection among that fixed candidate set.
    """
    rows=[]; selections=[]
    for name,tr,va in distributed_splits(data,cfg,folds=outer_folds):
        train, valid = subset(data,tr), summarize_runs(subset(data,va),cfg)
        result=develop(train,replace(cfg,seed=cfg.seed+len(selections)+100),recipes)
        selected=result["finalists"]["recipes"]
        bundle=fit_bundle(train,selected,cfg)
        pred=predict_bundle(bundle,valid[list(cfg.factors)].to_numpy(),cfg)
        rows.append(valid.assign(predicted=pred,outer_fold=name))
        selections.append(selected)
    frame=pd.concat(rows,ignore_index=True)
    return {"predictions":frame,"metrics":metrics(frame["mean"],frame.predicted,cfg.tolerance),"selections":selections}
