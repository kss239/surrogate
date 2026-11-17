"""Input/output-free audit, grouping, partitioning, and synthetic fixture functions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.model_selection import train_test_split


def scale_bounds(X, cfg):
    b = np.asarray(cfg.bounds, float)
    return (np.asarray(X, float) - b[:, 0]) / (b[:, 1] - b[:, 0])


def synthetic_runs(cfg, n_alternatives=180, repetitions=8):
    rng = np.random.default_rng(cfg.seed)
    z = qmc.LatinHypercube(len(cfg.factors), seed=cfg.seed).random(n_alternatives)
    b = np.asarray(cfg.bounds)
    X = b[:, 0] + z * (b[:, 1] - b[:, 0])
    # Preserve the original demonstration for >=8 factors; smaller studies use
    # only terms whose factors exist. Extra inputs may be irrelevant by design.
    d = len(cfg.factors)
    mu = 3 + 4*z[:, 0]
    if d >= 3: mu = mu + 9*(z[:, 1]-.5)*(z[:, 2]-.5)
    if d >= 4: mu = mu + 3*np.sin(np.pi*z[:, 3])
    if d >= 5: mu = mu + 2*z[:, 4]**2
    if d >= 7: mu = mu + 2*(z[:, 5] > .6)*z[:, 6]
    sigma = .25 + .9*z[:, min(7, d-1)]
    counts = np.asarray(repetitions)
    if counts.ndim == 0: counts = np.full(n_alternatives, counts.item())
    if counts.shape != (n_alternatives,) or not np.isfinite(counts).all() or np.any(counts < 1) or np.any(counts != counts.astype(int)):
        raise ValueError("Repetitions must be a positive integer or one positive count per alternative")
    counts = counts.astype(int)
    return pd.DataFrame(np.repeat(X, counts, axis=0), columns=cfg.factors).assign(
        **{cfg.alternative: np.repeat([f"Alt{i:04d}" for i in range(n_alternatives)], counts),
           cfg.repetition: np.concatenate([np.arange(n) for n in counts]),
           cfg.response: np.repeat(mu, counts) + np.repeat(sigma, counts)*rng.normal(size=counts.sum()),
           "_synthetic_true_mean": np.repeat(mu, counts)})



def load_runs(path, cfg):
    if path is None:
        return synthetic_runs(cfg)
    return pd.read_parquet(path) if Path(path).suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)


def audit_runs(df, cfg, feasibility):
    if len(cfg.factors) != len(cfg.bounds) or len(set(cfg.factors)) != len(cfg.factors):
        raise ValueError("Factor names and bounds must be unique and aligned")
    if cfg.response in cfg.factors or cfg.alternative in cfg.factors or cfg.repetition in cfg.factors:
        raise ValueError("IDs and response cannot be factors")
    if not (0 < cfg.test_fraction < 1 and 0 < cfg.calibration_fraction < 1
            and cfg.test_fraction + cfg.calibration_fraction < .7):
        raise ValueError("Invalid partition fractions")
    if not 0 < cfg.alpha < 1 or cfg.tolerance <= 0:
        raise ValueError("Invalid alpha or tolerance")
    cols = [cfg.alternative, cfg.repetition, *cfg.factors, cfg.response]
    missing = set(cols) - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {missing}")
    out = df.loc[:, cols].copy()
    if out.isna().any().any(): raise ValueError("Missing values: classify/fix before training")
    if out.duplicated([cfg.alternative, cfg.repetition]).any():
        raise ValueError("Duplicate alternative/repetition key")
    numeric = out[[*cfg.factors, cfg.response]].to_numpy(float)
    if not np.isfinite(numeric).all(): raise ValueError("Nonfinite data")
    b = np.asarray(cfg.bounds, float)
    if not np.isfinite(b).all() or np.any(b[:, 1] <= b[:, 0]): raise ValueError("Invalid bounds")
    X = numeric[:, :-1]
    if np.any(X < b[:, 0]) or np.any(X > b[:, 1]): raise ValueError("Factors outside physical bounds")
    if not np.asarray(feasibility(X), bool).all(): raise ValueError("Infeasible observed factors")
    if (out.groupby(cfg.alternative)[list(cfg.factors)].nunique() > 1).any().any():
        raise ValueError("An alternative changes factor values across repetitions")
    keys = list(map(tuple, X))
    mapping = {key: i for i, key in enumerate(sorted(set(keys)))}
    out = out.assign(group=[mapping[key] for key in keys])
    if out.empty: raise ValueError("No simulation observations provided")
    return out


def summarize_runs(runs, cfg):
    stats = runs.groupby("group")[cfg.response].agg(mean="mean", variance="var", reps="size")
    factors = runs.groupby("group")[list(cfg.factors)].first()
    return factors.join(stats).assign(mean_variance=lambda d: d.variance/d.reps).reset_index()


def split_groups(summary, cfg):
    groups = summary.group.to_numpy()
    support = np.array([], dtype=groups.dtype)
    if cfg.split_strategy == "boundary":
        from .boundary import boundary_support
        support = boundary_support(summary,cfg)['selected'].group.to_numpy()
        groups = groups[~np.isin(groups,support)]
    n_test = int(np.ceil(len(groups)*cfg.test_fraction))
    n_cal = int(np.ceil((len(groups)-n_test)*cfg.calibration_fraction/(1-cfg.test_fraction)))
    n_dev = len(groups)-n_test-n_cal
    if n_test < 1 or n_cal < 1 or n_dev < 3:
        raise ValueError("Too few alternatives for development/calibration/test partitions; need at least 3 development groups and 1 in each holdout. Adjust fractions or use audited data with an explicit validation split.")
    remaining, test = train_test_split(groups, test_size=cfg.test_fraction, random_state=cfg.seed)
    development, calibration = train_test_split(remaining,
        test_size=cfg.calibration_fraction/(1-cfg.test_fraction), random_state=cfg.seed+1)
    return {"development": np.sort(np.concatenate([development,support])), "calibration": np.sort(calibration), "test": np.sort(test)}


def subset(runs, groups):
    return runs.loc[runs.group.isin(groups)].copy()


def study_inventory(runs, cfg):
    """Observed study dimensions; no required alternative count or balanced replication."""
    counts = runs.groupby('group').size()
    return pd.DataFrame([dict(factors=len(cfg.factors), alternative_ids=runs[cfg.alternative].nunique(),
        unique_configurations=len(counts), total_runs=len(runs),
        min_repetitions=int(counts.min()), median_repetitions=float(counts.median()),
        max_repetitions=int(counts.max()), balanced_repetitions=bool(counts.nunique()==1),
        singleton_configurations=int((counts==1).sum()), replicated_configurations=int((counts>=2).sum()))])


def training_mean_noise(summary):
    """Training-only pooled run variance for singleton GP observations, never zero fill.

    Returns None when no replication is available; the GP must learn shared noise.
    Raw descriptive summaries retain NaN for unobserved within-group variance.
    """
    known = summary.reps >= 2
    if not known.any(): return None
    pooled = float((summary.loc[known, 'variance']*(summary.loc[known, 'reps']-1)).sum()
                   /(summary.loc[known, 'reps']-1).sum())
    return summary.variance.fillna(pooled).to_numpy()/summary.reps.to_numpy()
