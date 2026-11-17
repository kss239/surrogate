# Validation record — v0.4.0

Validated November 16, 2026 (America/Denver), with synthetic data only.

- **58 tests passed**, including all previous model/dependency tests plus arbitrary
  factor counts (1, 2, 5, 17), unequal and odd repetition counts, singleton
  configurations, discovered study dimensions, and removal of the fixed
  40-configuration audit restriction.
- Small-data checks exercised a five-alternative study, guarded validation splits,
  singleton mean-model fitting, and clear rejection when three-way partitioning
  is mathematically impossible. Small data acceptance does not imply sufficient
  evidence for accurate prediction.
- Tests checked unknown variances remain NaN; replicated-subset noise reporting;
  training-only pooled GP noise and learned shared GP noise; latent-mean SD;
  uncertainty/calibration without any repetitions; pilot replication proposals;
  one-factor interaction/coverage tables; and actual learning-curve run counts.
- **Three complete notebook executions passed**, all 12 code cells each:
  - Original example: 12 factors, 180 alternatives, 8 repetitions each.
  - One-factor example: 24 alternatives, 1 observation each.
  - Five-factor example: 24 alternatives, counts repeating 1, 2, 3, 7.
- Each notebook run completed **26 initial recipes / 78 grouped CV fits with zero
  failed initial trials**, plus selection, learning curves, uncertainty, proposals,
  final evaluation and post-model diagnostics. The default notebook's executed
  outputs are saved; variant executions were validation fixtures.
- The two smaller examples had five calibration alternatives and correctly
  returned unbounded 90% conformal intervals. The unreplicated example reported
  unknown separate run-noise SD and used absolute-residual calibration.
- A current **0.4.0 wheel** is built in `dist/`. Runtime dependencies are unchanged
  from the versions listed in `requirements-tested.txt`.

All existing caveats about independent simulation randomness, held-out evaluation,
continuous inputs/responses, and the advanced research tier still apply. No user
simulation data was supplied.

---

# Previous validation — v0.3.0

Validated November 15, 2026 (America/Denver), on synthetic data only, with
Python 3.13.0. Exact runtime versions are in `requirements-tested.txt`.

## Completed checks

- **47 tests passed**, including all nine core model families; optional pyGAM,
  EBM, XGBoost and LightGBM fits; GP RBF/Matérn ARD and scaled observation-noise
  checks; response-specific noise scaling; crossed allocation cost/counts;
  group-disjointness and validation-label perturbation; marginal/pairwise
  coverage; residual and training-threshold tail tables; flat prediction clouds;
  keyed prediction alignment; nested selection; calibration; checkpoints;
  end-to-end prediction, DOE proposals and export.
- The full notebook ran through a local Jupyter kernel: **12 code cells**, no
  execution errors. Its schema was validated and executed outputs saved.
- Default synthetic smoke workflow: 180 alternatives × 8 repetitions, 12 factors,
  **26 candidate pipelines / 78 grouped CV fits**, zero failed initial trials.
  Regional development challenges, bootstrap uncertainty, calibration, crossed
  allocation experiments and the dedicated post-model phase also ran.
- The selected synthetic ensemble combines selected-interaction histogram
  boosting with quadratic Elastic Net. Synthetic latent-mean test RMSE was about
  **0.706**; one-future-run coverage was **32/36 (88.9%)** at nominal 90%.
  These are demonstration results, not estimates for the user's simulator.
- The post-model four-panel run/mean prediction figure was visually inspected.
- A current **0.3.0 wheel** was built successfully in `dist/`.

## Optional dependencies

All requested optional integrations were available and fitted successfully:
pyGAM 0.12.0, InterpretML core 0.7.8, XGBoost 3.4.1 and LightGBM 4.7.0.
The first environment resolved pyGAM 0.9.1, which failed on the removed SciPy
sparse-matrix `.A` property. The package now requires pyGAM >=0.10 and explicitly
rejects older versions. Installing pyGAM 0.12.0 resolved SciPy 1.16.3, and the full
test suite then passed. No compatibility monkey patch is applied.

Unavailable imports/native libraries are reported by `optional_model_status`;
a test exercises this failure path. Optional families are disabled by default
and were tested separately from the default notebook search. Their full broad
search budgets were not run.

## Limits of validation

No user simulation data was supplied. Full broad-budget search, multiprocessing,
Optuna, correlated/common-random-number inference, and the research-tier models
listed in README were not validated in this update. The nested selection helper
was exercised with a reduced candidate set. Uncertainty assumptions, noisy mean
references, row-weighted run metrics, descriptive diagnostic cutoffs and repeated
use of development folds remain documented limitations. Test outcomes were not
used for model selection. Final-test-guided changes require fresh confirmation
alternatives.

## Standalone response exploration — November 16, 2026

Added `Response_Exploration.ipynb` and functional response-space helpers.
The complete notebook executed and saved all outputs; its three rendered figure
panels were visually reviewed. All **61 tests passed**, including unequal
replication/unknown variance, exact linear association/PCA, unit-scale invariance,
constant responses, missing values and duplicate run keys. Demonstration outputs
use synthetic correlated responses without planted discrete clusters. No user
response table was supplied. Initialization stability is not resampling stability
or proof of natural populations; these limitations are explicit in the notebook.

## Boundary-supported splitting update

Added response-independent geometry selection, capped support budgets, retained
support in development folds, and selected-point/coverage-gap diagnostics.
Synthetic comparison in `boundary_comparison.csv` uses the same generated data
and prespecified quadratic recipe: random empirical-location CV RMSE 0.810;
support-conditioned non-support-location CV RMSE 0.732. These evaluate different
locations and training allocations and do not establish superiority of a strategy.
Notebook instructions updated; saved outputs cleared to avoid stale results.

## Expanded response exploration

- Eight response-exploration tests passed, covering existing behavior plus curved
  dependence, reproducible resampling, collinear partial correlations, separation
  of within-run and between-configuration associations, all-noise DBSCAN,
  subsampling stability, and Pareto direction/tie handling.
- Executed all 15 notebook code cells sequentially in-process with IPython on the
  labeled synthetic data; captured tables and plots and validated notebook format.
  A separate nbclient kernel could not launch because local socket binding is
  restricted in this environment. In-process execution succeeded.
- Inspected the nine generated figures. No real simulation data was used.
- Optional Pareto calculations are unit tested; the notebook skips them by default
  until objective directions are supplied. No full surrogate-model test rerun was
  performed for this isolated exploration expansion.

## Simplified observed-extreme support

Replaced joint/corner and space-filling selection with exact observed minima and
maxima, configured factor order, deterministic tie reuse and group deduplication.
Tests cover ties, shuffled input reproducibility, zero/insufficient budgets,
explicit uncovered extremes, absence of interior additions, arbitrary dimensions,
unequal repetitions, and fold retention. The full test suite passed. Notebook
schema validated; notebook instructions updated without re-execution. Refreshed
`boundary_comparison.csv` for the simplified selector; earlier numerical comparisons
above describe the superseded selector. Random grouped and edge stress tests remain.
