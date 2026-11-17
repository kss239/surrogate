"""OOF residual summaries, learning curves, and interaction screening."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import summarize_runs, subset
from .features import fit_features
from .models import fit_bundle, predict_bundle
from .validation import metrics


def diagnostic_table(oof):
    frame = oof.assign(support_band=pd.qcut(oof.distance.rank(method="first"), 3, labels=["near", "middle", "far"]),
                       absolute_error=abs(oof.residual))
    return frame.groupby("support_band", observed=True).agg(alternatives=("group", "size"),
        mae=("absolute_error", "mean"), bias=("residual", "mean"), mean_reference_variance=("mean_variance", "mean"))


def learning_curves(data, recipes, split, cfg):
    _, train_ids, valid_ids = split
    if set(train_ids) & set(valid_ids): raise ValueError("Overlapping train/validation groups")
    experiments = allocation_learning_curves(data, recipes, (split,), cfg,
        fractions=(.4,.7,1.), repetitions=(1,2,3,5,7,10), seeds=(4,))
    means = experiments.loc[experiments.grain == 'alternative_mean']
    largest_cap = means.repetitions.max()
    largest_size = means.alternatives.max()
    alternatives = means.loc[means.repetitions == largest_cap].assign(axis='alternatives', amount=lambda d:d.alternatives)
    reps = means.loc[means.alternatives == largest_size].assign(axis='repetition_cap', amount=lambda d:d.repetitions)
    return pd.concat([alternatives,reps],ignore_index=True)


def interaction_screen(data, cfg):
    s = summarize_runs(data, cfg)
    state = fit_features(s[list(cfg.factors)].to_numpy(), s["mean"].to_numpy(), "selected", cfg, 8)
    return pd.DataFrame([{"factor_a": cfg.factors[j], "factor_b": cfg.factors[k], "screen_score": score}
                         for (j,k), score in state["ranking"]], columns=["factor_a", "factor_b", "screen_score"])

def paired_error_comparison(first, second, cfg, draws=1000):
    """Paired alternative bootstrap of squared-error difference; assumes independent alternatives.
    Negative difference favors first; not a post-selection significance certificate.
    """
    merged = first[["group", "mean", "predicted"]].merge(
        second[["group", "predicted"]], on="group", suffixes=("_first", "_second"), validate="one_to_one")
    delta = ((merged["mean"]-merged.predicted_first)**2 - (merged["mean"]-merged.predicted_second)**2).to_numpy()
    rng = np.random.default_rng(cfg.seed)
    sampled = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)])
    return {"mean_squared_error_difference": float(delta.mean()),
            "bootstrap_95_lower": float(np.quantile(sampled,.025)),
            "bootstrap_95_upper": float(np.quantile(sampled,.975)), "alternatives": len(delta)}


def response_noise_report(raw, responses, cfg, feasibility):
    """Audit each continuous response independently; equal-alternative noise estimates.

    Independent repetitions are required. Negative raw signal estimates are retained;
    clipped fractions are descriptive estimates, not attainable performance guarantees.
    """
    from dataclasses import replace
    from .data import audit_runs
    rows = []
    for response in responses:
        c = replace(cfg, response=response)
        s = summarize_runs(audit_runs(raw, c, feasibility), c)
        rows.append(dict(response=response, alternatives=len(s), runs=int(s.reps.sum()),
                         **noise_summary(s)))
    return pd.DataFrame(rows)


def allocation_learning_curves(data, recipes, splits, cfg, fractions=(.4, .7, 1.),
                               repetitions=(2, 3, 5, 7, 10), seeds=(0, 1, 2)):
    """Crossed, nested training budgets with fixed validation references per split.

    Repetitions are per-alternative caps: use min(cap, available), retaining unequal counts.
    Recipes must be prespecified or selected independently of these validation folds.
    Returns individual experiments, not confidence intervals from dependent subsamples.
    """
    if not fractions or any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("Fractions must be in (0, 1]")
    if not repetitions or any(r < 1 or int(r) != r for r in repetitions):
        raise ValueError("Repetition caps must be positive integers")
    rows = []
    for name, tr, va in splits:
        if set(tr) & set(va): raise ValueError("Overlapping train/validation groups")
        base, held = subset(data, tr), subset(data, va)
        valid = summarize_runs(held, cfg)
        max_reps = int(base.groupby('group').size().max())
        counts = sorted(set([r for r in repetitions if r <= max_reps] + [max_reps]))

        for seed in seeds:
            rng = np.random.default_rng(cfg.seed+seed)
            order = rng.permutation(tr)
            shuffled = {g: block.iloc[rng.permutation(len(block))] for g, block in base.groupby('group')}
            sizes = sorted(set(min(len(order), max(2, int(len(order)*f))) for f in fractions))
            for size in sizes:
                for count in counts:
                    train = pd.concat([shuffled[g].iloc[:count] for g in order[:size]])
                    actual_counts = train.groupby('group').size()
                    bundle = fit_bundle(train, recipes, cfg)
                    pred = predict_bundle(bundle, valid[list(cfg.factors)].to_numpy(), cfg)
                    joined = held.merge(valid[['group']].assign(predicted=pred), on='group', validate='many_to_one')
                    for grain, y, p in [('alternative_mean', valid['mean'], pred),
                                        ('individual_run', joined[cfg.response], joined.predicted)]:
                        rows.append(dict(split=name, seed=seed, alternatives=size, repetitions=count,
                            min_repetitions=int(actual_counts.min()), max_repetitions=int(actual_counts.max()),
                            mean_repetitions=float(actual_counts.mean()),
                            simulation_runs=len(train), simulation_cost=len(train)*cfg.cost_per_run,
                            grain=grain, **metrics(y, p, cfg.tolerance)))
    return pd.DataFrame(rows)


def coverage_report(train, held, cfg, bins=5):
    """Physical-bound scaled coverage; box discrepancy is not feasible-domain coverage."""
    from itertools import combinations
    from scipy.stats import qmc
    from sklearn.neighbors import NearestNeighbors
    from .data import scale_bounds
    from .validation import support_distance
    X = scale_bounds(train[list(cfg.factors)].to_numpy(), cfg)
    nearest = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    marginal = pd.DataFrame([dict(factor=f, unique_values=len(np.unique(X[:, j])),
        scaled_span=float(np.ptp(X[:, j])), occupied_bin_fraction=float(len(np.unique(np.minimum((X[:, j]*bins).astype(int), bins-1)))/bins))
        for j, f in enumerate(cfg.factors)])
    pairs = []
    cells = np.minimum((X*bins).astype(int), bins-1)
    for j, k in combinations(range(X.shape[1]), 2):
        pairs.append(dict(factor_a=cfg.factors[j], factor_b=cfg.factors[k],
                          occupied_cell_fraction=len(np.unique(cells[:, [j,k]], axis=0))/bins**2))
    distances = support_distance(held[list(cfg.factors)].to_numpy(), train[list(cfg.factors)].to_numpy(), cfg)
    return dict(marginal=marginal, pairwise=pd.DataFrame(pairs, columns=["factor_a", "factor_b", "occupied_cell_fraction"]),
                summary=pd.DataFrame([dict(alternatives=len(X), dimensions=X.shape[1],
                    quadratic_coefficients=(X.shape[1]+1)*(X.shape[1]+2)//2,
                    box_centered_discrepancy=qmc.discrepancy(X),
                    nearest_neighbor_median=float(np.median(nearest)),
                    heldout_distance_p90=float(np.quantile(distances, .9)))]),
                heldout_distances=pd.DataFrame({'group':held.group.to_numpy(), 'distance':distances}))


def residual_patterns(means, cfg):
    """Descriptive residual bins; no multiple-testing or causal interpretation."""
    frame = means.assign(residual=means['mean']-means.predicted)
    rows=[]
    for column in [*cfg.factors, 'predicted', 'distance']:
        if column not in frame: continue
        bands = pd.qcut(frame[column], 4, duplicates='drop')
        for band, block in frame.groupby(bands, observed=True):
            rows.append(dict(variable=column, band=str(band), alternatives=len(block),
                bias=float(block.residual.mean()), rmse=float(np.sqrt(np.mean(block.residual**2))),
                mean_reference_variance=float(block.mean_variance.mean())))
    return pd.DataFrame(rows)


def tail_diagnostics(means, training_means, cfg):
    """Observed response tails use thresholds fixed from training means only."""
    lo, hi = np.quantile(training_means, [.1, .9])
    masks = [('low', means['mean'] <= lo), ('middle', (means['mean'] > lo)&(means['mean'] < hi)),
             ('high', means['mean'] >= hi)]
    rows=[]
    for label, mask in masks:
        block=means.loc[mask]
        row=dict(tail=label, alternatives=len(block), lower_threshold=lo, upper_threshold=hi)
        if len(block):
            row.update(metrics(block['mean'], block.predicted, cfg.tolerance))
        rows.append(row)
    return pd.DataFrame(rows)


def noise_summary(summary):
    """Descriptive decomposition only when all alternative variances are observed.

    Replicated-subset summaries remain useful with singleton alternatives, but
    cannot establish the overall noise/signal decomposition without assumptions.
    """
    known = summary.reps >= 2
    complete = bool(known.all())
    between = float(summary['mean'].var(ddof=1))
    mean_noise = float(summary.mean_variance.mean()) if complete else np.nan
    run_noise = float(summary.variance.mean()) if complete else np.nan
    raw_signal = between-mean_noise
    signal = max(0.,raw_signal) if np.isfinite(raw_signal) else np.nan
    return dict(noise_status="estimated_from_repetitions" if complete else "unavailable_for_singletons",
        replicated_alternatives=int(known.sum()), singleton_alternatives=int((~known).sum()),
        between_mean_variance=between, mean_sampling_variance=mean_noise,
        run_noise_variance=run_noise, signal_variance_raw=raw_signal, signal_variance=signal,
        estimated_run_signal_fraction=signal/(signal+run_noise) if signal+run_noise > 0 else np.nan,
        estimated_mean_signal_fraction=signal/(signal+mean_noise) if signal+mean_noise > 0 else np.nan,
        replicated_subset_average_run_variance=float(summary.loc[known,'variance'].mean()),
        median_mean_se=float(np.sqrt(summary.loc[known,'mean_variance']).median()),
        variance_p90=float(summary.loc[known,'variance'].quantile(.9)),
        variance_p10=float(summary.loc[known,'variance'].quantile(.1)))
