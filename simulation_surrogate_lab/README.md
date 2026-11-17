# Simulation Surrogate Lab

A functional Python package plus one orchestration notebook for predicting a selected response from continuous simulation factors and repeated runs.

## Install and run
From this extracted project directory:
```bash
python -m pip install -e ".[notebook,test,parquet]"
python -m pytest
jupyter lab Surrogate_Workflow.ipynb
```
Optional optimizers: `python -m pip install -e ".[optimization]"`.
Build a current wheel with `uv build --wheel`. A current 0.4.0 wheel is included in `dist/`; use it or the editable source instead of older wheels.

For a minimal environment without Jupyter: `python -m pip install -e .`.

The notebook starts with labeled synthetic data and small **smoke** budgets. Edit the data path, factor names, physical bounds, selected response, feasibility rule, and error tolerance before using actual simulation data. Set `profile="broad"` after the smoke run succeeds. Every response has an independent study.


## Configure a general study (v0.4)

The factor count, alternative count and repetition count are **not fixed**.
Twelve factors and balanced repetitions are demonstration choices only.
Define the continuous input columns and their physical bounds:

```python
from surrogate_lab import config_from_bounds
from surrogate_lab.workflow import prepare_study

cfg = config_from_bounds(
    {"temperature": (20., 100.), "pressure": (1., 10.), "flow": (0.1, 5.)},
    response="output", alternative="alternative_id", repetition="run_id",
    tolerance=0.5,
)
study = prepare_study(raw_dataframe, cfg)
print(study["inventory"].to_string(index=False))
```

The mapping may have one, three, seventeen, or another number of factors. Do not
enter the number of alternatives or repetitions. The input rows determine them.
The inventory reports input count, alternative IDs, unique configurations, total
runs, minimum/median/maximum repetitions, balance, and singleton counts. Individual
counts are available in `study["summary"][["group", "reps"]]`. Odd counts and
unequal counts are both acceptable. Duplicate factor vectors still share a group.
For compatibility, `Config()` retains the old demonstration defaults; use
`config_from_bounds` or supply both `factors` and `bounds` for actual studies.
Bounds are supplied domain limits, not inferred using held-out observations.

### What changes when some alternatives have one run?

- All observations remain eligible for mean-model fitting and grouped validation.
  A one-run sample mean is simply that observation, not a precise expected response.
- Its sample variance and mean sampling variance remain **unknown (`NaN`)**.
  Whole-study signal/noise decomposition is unavailable if any reference
  alternative lacks replication. Replicated-subset summaries are labeled.
- For GP fitting, singleton noise uses the pooled within-alternative variance from
  that training fold's replicated alternatives. This is an explicit approximation,
  not a measured local variance. Without replication, a shared noise kernel is
  learned; `noise_method` identifies the approach. The GP mean-SD helper removes
  this learned observation-noise component from its returned latent-mean SD.
- The separate variance model trains only on replicated alternatives. If there
  are none, `estimated_run_sd` is unavailable and future-run conformal intervals
  use absolute residuals (constant calibration scale of 1). This gives intervals
  without claiming to separate stochastic noise from model error.
- New-run proposals prioritize obtaining repetitions at singleton alternatives.
  Their predicted local variance reduction is left unknown.

Mean metrics still give each configuration equal weight; individual-run metrics
weight each observed row. Weighted-run fitting gives each alternative equal total
training weight, regardless of its number of repetitions.

### Dataset size and learning experiments

The audit accepts small datasets. Development/calibration/test splitting requires
at least three development configurations and one in each holdout, after applying
the requested fractions. Grouped CV reduces excessive fold counts to the available
number of groups and requires two training configurations per fold. Regional
holdouts retain at least two training configurations. Smaller data can be audited
and fitted directly, but cannot support every validation operation. Small
calibration sets may correctly produce unbounded intervals.

Learning experiments use `min(repetition_cap, available_runs)` for each selected
configuration. They never invent observations or reduce every alternative to the
smallest available repetition count. The largest cap includes all available runs.
`simulation_runs`, `simulation_cost`, and actual min/mean/max repetition counts
make unequal budgets explicit. One-factor studies omit pairwise surface plots and
return empty, correctly structured pairwise interaction/coverage tables.

## Modules and functional design
| Module | Responsibility |
|---|---|
| config | Frozen configuration and recipe records |
| data | Loading, strict audits, duplicate-configuration grouping, splitting, synthetic fixture |
| features | Fold-local representations and interaction screening |
| models | Fresh estimator factories, fit, prediction |
| validation | Common distributed/regional splits, metrics, OOF evaluation |
| search | Branch-balanced random search, checkpoints, refinement, optional TPE |
| selection | Development-only finalists, ensemble selection, seed stability |
| diagnostics | Response-specific noise, crossed allocation curves, DOE coverage, residual/tail summaries |
| postdiagnostics | Dedicated post-model investigation and descriptive findings |
| uncertainty | Approximate bootstrap spread, variability model, future-run conformal calibration |
| design | Diverse new alternatives and additional-repetition proposals |
| plots | Figures returned to caller, no file writes |
| storage | Source hashing, explicit artifact export |
| workflow | Preparation, development search and independent evaluation orchestration |

Core computations receive data and configuration explicitly. Functions return new results and do not mutate input tables. Configuration/recipes are immutable; third-party estimators are created and fitted locally. Estimators are internally mutable, so this is practical functional design, not mathematical purity. Disk writes occur only at explicit search/export boundaries. No simulator or external service is invoked.

## Data contract
One row per alternative/repetition; all configured continuous factors plus one selected continuous response. Valid zeros are retained. IDs and other responses are excluded as predictors. Each alternative must have constant factors; identical factor vectors under distinct IDs share one group. One or more observations per unique configuration are accepted; repetition counts may differ arbitrarily. Missing/nonfinite data, duplicate keys, out-of-range or infeasible points fail explicitly. There is no fixed alternative-count requirement at the audit stage. Each validation operation checks that its partitions contain enough groups to run; this is a computational requirement, not a claim of statistical adequacy. Use one coherent scenario/version, or explicitly extend the model and splitting scheme.

## Statistical boundaries
- All supervised feature discovery is inside training folds.
- Development, calibration, and final-test groups are disjoint. The predictor is frozen before calibration and never refit using test outcomes.
- Selection/CV scores are tuning scores; final-test metrics evaluate the frozen procedure. Repeated test-guided revision requires new confirmation data.
- Distributed and regional validation answer different questions. The configured regional penalty is a declared selection heuristic.
- Mean scores use noisy repetition means. Bootstrap spread is approximate, not a guaranteed latent-mean confidence interval.
- Calibrated intervals target ONE FUTURE RUN at a new alternative, not its latent mean. Marginal conformal guarantees require exchangeable units. Fixed/adaptive DOE and common random numbers may violate assumptions. Correlated repetitions require extensions.
- Interaction screening is a residual-product heuristic, not SHAP or a causal test. Full quadratic and nonlinear raw-factor branches remain in the search.
- Adaptive acquisition combines uncertainty, coverage, and predicted OOF error heuristically; replication proposals use local estimated sampling-variance reduction. This is not a proven globally optimal allocation.

## Search and persistence
Nine core families: ridge, Elastic Net, kernel ridge (RBF), random forest, Extra Trees, histogram boosting, SVR, MLP, Gaussian process. Optional GAM, EBM, XGBoost and LightGBM are explicitly enabled in configuration. Original, quadratic, additive-spline, selected-product, and approved-domain representations are supported in compatible branches. Mean and equal-alternative-weighted run fitting are compared where supported. Targets use affine scaling only.

Checkpoints contain data/source/config/split/runtime identity and atomic per-trial JSON files. Re-running resumes identical work. Time limits are soft: active fits finish. `workers` controls process concurrency; model threads are limited. Broad mode is expensive; inspect the planned trial table first. Search failures are recorded distinctly from weak scores. Trial and split provenance accompany exports. Only load trusted model artifacts.

## Scope
This release implements a working foundation and demonstration, not every extension discussed. Optional full nested evaluation is provided as a function but not run by default. No automatic simulator execution, covariance-aware shared-seed uncertainty, SHAP/Sobol analysis, or hard distributed-job timeouts. See `VALIDATION.md` for checks actually run.

## Post-model investigation (v0.2)
`postdiagnostics.py` provides `postfit_report` and configurable `DiagnosticThresholds`. Notebook phase 10B compares training and held-out predictions at run and alternative-mean grains, detects near-flat/compressed predictions, checks performance against the training mean baseline, inspects transformed inputs, and reports a noise decomposition with explicit assumptions. No retuning occurs. A keyed optional prediction table verifies factor alignment; upstream label correctness still requires provenance checks.


## Expanded framework (v0.3)

Regularized linear and full quadratic surfaces are available through both `ridge` and
`elastic` recipes. `ridge` with `spline` representation is a core additive spline
baseline; optional `gam` uses pyGAM's penalized smooths. GP search compares Matérn
and RBF kernels, with isotropic or per-factor ARD length scales. GP observation
noise for replicated alternatives is the estimated variance of each repetition mean,
scaled consistently with the target. Singletons use a training-only pooled run variance when replication exists; without replication the GP learns a shared WhiteKernel noise term. This is a plug-in heteroscedastic GP when replicated, not
joint noise-process inference. Exact GP is capped by `max_exact_gp_alternatives`.

| Optional family | Install | Configuration |
|---|---|---|
| pyGAM | `pip install -e '.[additive]'` | `enable_gam=True` |
| InterpretML EBM | `pip install -e '.[additive]'` | `enable_ebm=True` |
| XGBoost | `pip install -e '.[boosting]'` | `enable_xgboost=True` |
| LightGBM | `pip install -e '.[boosting]'` | `enable_lightgbm=True` |

`optional_model_status(cfg)` reports requested imports, versions, native-library
errors and install extras. Disabled packages are not imported. Automatic search
omits unavailable requested families and returns the status table under
`develop(...)["optional_models"]`; explicit failed recipes remain in the trial
ledger. An import success does not guarantee fit compatibility: fit failures are
also recorded. Optional package versions participate in checkpoint identity.
GAM, EBM and LightGBM train on alternative means. This also keeps EBM's internal
validation within the outer training alternatives. See the upstream
[pyGAM API](https://pygam.readthedocs.io/en/stable/reference/_autosummary/pygam.pygam.LinearGAM.html)
and [EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html).

All candidates use identical grouped development folds. Preprocessing, response
scaling, supervised interaction selection and GP noise estimates use only each
fold's training data. Search scores are optimistic after selection: call
`nested_evaluate` to rerun the entire selection process inside outer development
folds, or use the untouched final test for the frozen procedure. Neither the
post-model report nor the final interval diagnostics retunes models.

### Diagnostic outputs

- `response_noise_report(raw, responses, cfg, feasibility)` audits each response
  independently and reports within-alternative noise, mean sampling variance,
  unclipped/clipped signal estimates, estimated signal fractions, and mean SE.
  Use development rows before selection. Variances scale with response units;
  fractions are descriptive estimates, not hard R² ceilings.
- Search records expose `metrics` for equal-weight alternative means and
  `run_metrics` for individual repetitions. Run metrics weight rows, so alternatives
  with more repetitions contribute more; this differs from equal-alternative
  training weights. Final evaluation reports both grains too.
- `allocation_learning_curves` crosses alternative fractions with repetition
  caps across seeds and grouped splits. Unequal counts are retained, with actual counts/costs reported. Training subsets are nested, validation
  means retain all available repetitions, and each row records simulation count
  and cost. Use prespecified recipes or independently selected recipes. The older
  `learning_curves` helper remains for quick one-axis plots. Shared validation
  sets make curve points dependent; seed SD is not a confidence interval.
- `postfit_report` compares training and held-out spread, baseline skill and bias,
  flags horizontal prediction clouds and possible underfitting, and reports
  response noise, transformed-input checks, marginal/pairwise DOE occupancy,
  nearest-neighbor distances, box discrepancy, residual bins and tail errors.
  Tail thresholds are fixed from training means. Coverage uses physical bounds;
  empty bins can be legitimate under feasibility constraints. It cannot certify
  joint high-dimensional coverage from good one-dimensional projections.
- Final uncertainty output includes coverage versus nominal level and support-band
  coverage/count/width. Intervals target one future repetition. Bootstrap spread
  and GP latent-mean SD are separate approximate mean-uncertainty diagnostics,
  not calibrated mean confidence intervals. Small tails/regions remain unstable.

The notebook runs these as explicit phases and keeps final-test investigation
separate from development. Independent Monte Carlo repetitions are assumed;
shared seeds require covariance-aware extensions. No real simulation data was
provided for this change.

### Advanced research tier

CatBoost, NGBoost/distributional regression, quantile forests, sparse polynomial
chaos, RBF interpolation, BART, Bayesian neural networks, multitask GPs, mixture
of experts, symbolic regression and learned stacking are **not implemented**.
They require separate dependency, likelihood, tuning and validation decisions.
The existing MLP and development-selected equal-weight ensemble remain available.
Multiresponse noise comparison is implemented; jointly trained response models
and multiresponse acquisition are future work. The default workflow is for
continuous responses; count/binary likelihoods are not silently substituted.

Validation found all four optional families available. pyGAM requires >=0.10; the tested environment uses 0.12.0. See `VALIDATION.md` for the compatibility issue resolved during testing and exact validation scope.

## Standalone response exploration

Open `Response_Exploration.ipynb` to investigate relationships among multiple
continuous responses, independently of surrogate fitting. Set `DATA_PATH`,
`RESPONSES`, `GROUP_COLUMNS`, and optionally `REPETITION_COLUMN`. CSV and Parquet
are supported; the default is explicitly labeled synthetic data.

The notebook aggregates equally weighted configuration means, reports repetition
precision, excludes constant responses from PCA/clustering, compares Pearson and
Spearman associations, performs standardized PCA, and explores K-means partitions
in full response space. Cluster sizes, silhouette and initialization agreement
are descriptive; a cluster count is never automatically declared correct.
A replicated-only correlation comparison highlights one sensitivity check.
Missing/nonfinite values fail explicitly. Group by factors when duplicate IDs
represent the same physical configuration; otherwise use alternative IDs.
If factor grouping combines independent batches whose repetition labels overlap,
supply a globally unique run key or leave `REPETITION_COLUMN=None` after checking
source-key uniqueness separately.

Use development-only observations if findings will influence model selection.
Response-derived cluster labels are unavailable for new alternatives and must
not be joined into predictors of those same responses. No changes are made to
surrogate fitting or its saved results. The helper functions live in
`response_exploration.py`; this notebook needs no new dependencies.

## Optional observed-edge-supported validation

Set `split_strategy="boundary"` before examining test outcomes. Random grouped
splitting remains the default. The selector uses raw factor values only: visit
factors in configured order, minimum then maximum. For ties, reuse an already
nominated alternative if possible; otherwise choose the smallest audited group
ID (IDs follow sorted factor vectors). Deduplicate nominations. No scaling,
joint-corner targets, space-filling additions, or response values are used.

`boundary_budget` and `boundary_max_fraction` cap the selection. The existing
small-data safeguard reserves at least five non-support configurations. When
these limits cannot accommodate the nominations, retain the first nominations
in the documented order and report `budget_conflict`, `budget_limits`,
`required_edge_alternatives`, and `uncovered_extremes`. No interior points are
added to exhaust an unused budget. `boundary_tolerance` is retained for backward
configuration compatibility but is unused by this exact-extreme strategy.

`study['boundary']['selected']` lists groups, factors, and all extremes each
selected group covers. `coverage` lists every factor minimum/maximum, tied
candidates, selected supporting groups and distance from physical bounds.
`partition_sizes` reports resulting counts. Observed extrema can fall short of
physical bounds. Being inside every marginal range does not guarantee joint
support or interpolation.

Selected groups remain in development and in every ordinary development CV
training fold, with every repetition together. Apply the planned grouped split
to the remaining alternatives; fractions therefore apply to the remaining pool.
Edge/cluster stress tests deliberately remove support and remain separate tests.
Nested evaluation retains the originally selected geometry-only support in outer
and inner training folds; scores target non-support locations conditional on
that support. Calibration/test sets randomly sample remaining configurations;
provide independent near-edge alternatives when predictions there matter.
Choose the strategy before final-test outcomes are inspected. Feature fitting
and tuning remain restricted to training-fold outcomes.
