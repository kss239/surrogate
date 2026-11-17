"""Conditional recipe generation, resumable trials, and optional TPE refinement."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json, time
from importlib import import_module
from importlib.metadata import version
import numpy as np
import pandas as pd
import scipy, sklearn
from joblib import Parallel, delayed
from . import __version__ as ENGINE_VERSION
from .config import Recipe
from .storage import stable_hash, data_hash
from .validation import evaluate_recipe


def sample_params(family, rng):
    log = lambda lo, hi: float(10**rng.uniform(lo, hi))
    if family == "elastic": return {"alpha": log(-4, 1), "l1_ratio": float(rng.choice([.1, .5, .9]))}
    if family == "kernel_ridge": return {"alpha": log(-4, 1), "gamma": log(-3, 0)}
    if family == "gam": return {"lam": log(-2, 3), "n_splines": int(rng.choice([5, 8, 12]))}
    if family == "ebm": return {"interactions": int(rng.choice([0, 4, 8])), "max_bins": 64, "max_rounds": 1000}
    if family == "lgbm": return {"num_leaves": int(rng.choice([7, 15, 31])), "n_estimators": 200,
                                  "learning_rate": log(-2, -.5), "min_child_samples": 10}
    if family == "ridge": return {"alpha": log(-3, 3)}
    if family in ("forest", "extra"):
        return {"min_samples_leaf": int(rng.choice([1, 2, 4, 8])), "max_features": float(rng.choice([.6, 1.]))}
    if family == "boost":
        return {"learning_rate": log(-1.6, -.6), "max_iter": int(rng.choice([100, 200, 350])),
                "max_leaf_nodes": int(rng.choice([7, 15, 31])), "min_samples_leaf": int(rng.choice([5, 10, 20])),
                "l2_regularization": log(-3, 1)}
    if family == "svr": return {"C": log(-.2, 2), "gamma": log(-3, -.2), "epsilon": log(-3, -.6)}
    if family == "mlp": return {"hidden_layer_sizes": (int(rng.choice([24, 48, 80])),), "alpha": log(-5, 0),
                                "learning_rate_init": log(-3.5, -2)}
    if family == "gp": return {"length_scale": log(-.5, .6), "nu": float(rng.choice([1.5, 2.5])), "kernel": str(rng.choice(["matern", "rbf"])), "ard": bool(rng.integers(2))}
    if family == "xgb": return {"max_depth": int(rng.choice([2, 4, 6])), "n_estimators": 200,
                                 "learning_rate": log(-2, -.5), "reg_lambda": log(-2, 2)}
    raise ValueError(family)


def compatible_branches(cfg, n_alternatives):
    branches = {
        "elastic": ["raw", "quadratic"], "kernel_ridge": ["raw"],
        "ridge": ["raw", "quadratic", "spline", "selected"],
        "forest": ["raw", "selected"], "extra": ["raw", "selected"],
        "boost": ["raw", "selected"], "svr": ["raw"], "mlp": ["raw"],
        "gp": ["raw"] if n_alternatives <= cfg.max_exact_gp_alternatives else []}
    for row in optional_model_status(cfg).to_dict("records"):
        if row["enabled"] and row["available"]:
            branches[row["family"]] = ["raw"]
    if cfg.domain_features:
        branches = {k: v+["domain"] if v and k != "gp" else v for k, v in branches.items()}
    return tuple((family, rep, mode) for family, reps in branches.items() for rep in reps
                 for mode in (("mean", "run") if family in ("ridge", "forest", "extra", "boost", "xgb") else ("mean",)))


def make_recipes(cfg, n_alternatives, extra_per_family=None):
    rng = np.random.default_rng(cfg.seed)
    branches = compatible_branches(cfg, n_alternatives)
    recipes = [Recipe(f, r, mode, tuple(sorted(sample_params(f, rng).items())),
                      int(rng.choice([4, 8, 16])), cfg.seed) for f, r, mode in branches]
    extra = (0 if cfg.profile == "smoke" else 25) if extra_per_family is None else extra_per_family
    for family in sorted(set(b[0] for b in branches)):
        eligible = [b for b in branches if b[0] == family]
        for _ in range(extra):
            f, r, mode = eligible[int(rng.integers(len(eligible)))]
            recipes.append(Recipe(f, r, mode, tuple(sorted(sample_params(f, rng).items())),
                                  int(rng.choice([4, 8, 16, 24])), cfg.seed))
    return tuple(dict.fromkeys(recipes))


def execute_trial(recipe, data, splits, cfg, directory):
    trial_id = stable_hash(asdict(recipe))
    path = Path(directory)/f"{trial_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.resume and path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    try:
        result = evaluate_recipe(recipe, data, splits, cfg)
    except Exception as exc:
        result = {"status": "failed", "recipe": asdict(recipe), "seconds": time.perf_counter()-started,
                  "error": f"{type(exc).__name__}: {exc}"}
    result = json.loads(json.dumps({"trial_id": trial_id, **result}, default=str))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, default=str))
    temporary.replace(path)
    return result


def run_search(recipes, data, splits, cfg, fingerprint):
    scope = stable_hash({"data": data_hash(data), "config": asdict(cfg), "splits": [(n, tr.tolist(), va.tolist()) for n,tr,va in splits],
                         "source": fingerprint, "engine": ENGINE_VERSION, "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "optional": optional_model_status(cfg).to_dict("records")})
    directory = Path(cfg.checkpoint_dir)/scope
    directory.mkdir(parents=True, exist_ok=True)
    started, results = time.perf_counter(), []
    workers = max(1, cfg.workers)
    for offset in range(0, len(recipes), workers):
        if time.perf_counter()-started > cfg.soft_seconds: break
        batch = recipes[offset:offset+workers]
        new = Parallel(n_jobs=workers)(delayed(execute_trial)(r, data, splits, cfg, directory) for r in batch)
        results.extend(new)
    return {"records": tuple(results), "directory": str(directory), "planned": len(recipes),
            "unstarted": len(recipes)-len(results), "seconds": time.perf_counter()-started}


def leaderboard(records):
    rows = [{"trial_id": r["trial_id"], "family": r["recipe"]["family"],
             "representation": r["recipe"]["representation"], "training": r["recipe"]["training"],
             **r["metrics"], "fold_rmse_sd": r["fold_rmse_sd"], "seconds": r["seconds"],
             "warning_count": len(r["warnings"])} for r in records if r["status"] == "complete"]
    if not rows: raise RuntimeError("No successful trials. Inspect search records.")
    return pd.DataFrame(rows).sort_values(["rmse", "seconds"]).reset_index(drop=True)


def restore_recipe(record):
    d = record["recipe"]
    params = tuple((k, tuple(v) if isinstance(v, list) else v) for k,v in d["params"])
    return Recipe(d["family"], d["representation"], d["training"], params, d["top_pairs"], d["seed"])


def refinement_recipes(records, cfg, per_branch=5, top_branches=4):
    board = leaderboard(records).drop_duplicates(["family", "representation", "training"]).head(top_branches)
    rng = np.random.default_rng(cfg.seed+90)
    return tuple(Recipe(row.family, row.representation, row.training,
                        tuple(sorted(sample_params(row.family, rng).items())),
                        int(rng.choice([4, 8, 16, 24])), cfg.seed)
                 for row in board.itertuples() for _ in range(per_branch))


def optuna_refine(base, data, splits, cfg, fingerprint, trials=30):
    """Optional TPE refinement of one branch; uses development folds only."""
    import optuna
    def objective(trial):
        family = base.family
        if family == "ridge": params = {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
        elif family in ("forest", "extra"):
            params = {"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
                      "max_features": trial.suggest_float("max_features", .4, 1.)}
        elif family == "svr":
            params = {"C": trial.suggest_float("C", .5, 200, log=True),
                      "gamma": trial.suggest_float("gamma", .001, 1, log=True),
                      "epsilon": trial.suggest_float("epsilon", .001, .3, log=True)}
        elif family == "boost":
            params = {"learning_rate": trial.suggest_float("learning_rate", .02, .25, log=True),
                      "max_iter": trial.suggest_int("max_iter", 100, 500, step=100),
                      "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 40),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25),
                      "l2_regularization": trial.suggest_float("l2_regularization", .001, 10, log=True)}
        else: raise ValueError("Optuna helper supports ridge, forest, extra, svr, boost; other families use the core search")
        recipe = replace(base, params=tuple(sorted(params.items())))
        result = run_search((recipe,), data, splits, cfg, fingerprint)["records"][0]
        trial.set_user_attr("trial_id", result["trial_id"])
        return result["metrics"]["rmse"] if result["status"] == "complete" else float("inf")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=trials, timeout=cfg.soft_seconds)
    return study


def optional_model_status(cfg):
    """Probe requested optional imports, including native-library failures; never hide omissions."""
    rows = []
    for family, module, flag, extra in (("xgb", "xgboost", "enable_xgboost", "boosting"),
            ("lgbm", "lightgbm", "enable_lightgbm", "boosting"),
            ("gam", "pygam", "enable_gam", "additive"),
            ("ebm", "interpret", "enable_ebm", "additive")):
        enabled = getattr(cfg, flag)
        available, installed, reason = False, None, "disabled (not probed)"
        if enabled:
            try:
                import_module(module)
                installed = version("interpret-core" if module == "interpret" else module)
                if module == "pygam" and tuple(int(x) for x in installed.split('.')[:2]) < (0, 10):
                    raise ImportError("pyGAM >=0.10 required for modern SciPy compatibility")
                available, reason = True, "available"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        rows.append(dict(family=family, enabled=enabled, available=available,
                         version=installed, reason=reason, install_extra=extra))
    return pd.DataFrame(rows)
