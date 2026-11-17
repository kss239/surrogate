"""Immutable study configuration and candidate recipes."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Config:
    # Backward-compatible demonstration defaults; supply your own factors and bounds.
    factors: tuple = tuple(f"factor_{i}" for i in range(1, 13))
    bounds: tuple = tuple((0., 1.) for _ in range(12))
    response: str = "response_1"
    alternative: str = "alternative"
    repetition: str = "repetition"
    split_strategy: str = "random"
    boundary_budget: int = 12
    boundary_max_fraction: float = .2
    boundary_tolerance: float = .15
    profile: str = "smoke"
    seed: int = 2026
    folds: int = 3
    test_fraction: float = .20
    calibration_fraction: float = .20
    tolerance: float = 1.0
    alpha: float = .10
    workers: int = 1
    soft_seconds: float = 3600.
    checkpoint_dir: str = "surrogate_checkpoints"
    resume: bool = True
    max_exact_gp_alternatives: int = 1200
    # Explicitly approved (operation, factor index, factor index) features.
    domain_features: tuple = ()
    denominator_floor: float = 1e-8
    # Change this identifier whenever the feasibility rule changes.
    feasibility_id: str = "box_only_v1"
    bootstrap_members: int = 8
    regional_penalty: float = .2
    enable_xgboost: bool = False
    enable_lightgbm: bool = False
    enable_gam: bool = False
    enable_ebm: bool = False
    candidate_count: int = 2048
    new_alternatives: int = 12
    new_repetitions: int = 5
    repeat_alternatives: int = 6
    additional_repetitions: int = 5
    cost_per_run: float = 1.0
    export_results: bool = False

    def __post_init__(self):
        if self.split_strategy not in ("random", "boundary") or self.boundary_budget < 0 or not 0 < self.boundary_max_fraction < .5 or not 0 <= self.boundary_tolerance <= 1:
            raise ValueError("Invalid boundary splitting configuration")
        if not self.factors or len(self.factors) != len(self.bounds) or len(set(self.factors)) != len(self.factors):
            raise ValueError("Provide at least one unique continuous factor and one bound pair per factor")
        if len({self.response, self.alternative, self.repetition, *self.factors}) != len(self.factors)+3:
            raise ValueError("Factor, response and identifier names must be distinct")
        if 'group' in (self.response, self.alternative, self.repetition, *self.factors):
            raise ValueError("The column name 'group' is reserved for internal configuration groups")
        if self.profile not in ("smoke", "broad"):
            raise ValueError("profile must be smoke or broad")
        if self.workers < 1 or self.folds < 2 or self.bootstrap_members < 2:
            raise ValueError("workers >= 1, folds >= 2, bootstrap_members >= 2 required")
        if self.candidate_count < self.new_alternatives or self.new_alternatives < 1:
            raise ValueError("candidate_count must cover positive new_alternatives")
        if min(self.new_repetitions, self.additional_repetitions, self.repeat_alternatives) < 1:
            raise ValueError("replication and repeat budgets must be positive")
        if self.soft_seconds <= 0 or self.cost_per_run <= 0 or self.regional_penalty < 0:
            raise ValueError("invalid time, cost, or regional penalty")

@dataclass(frozen=True)
class Recipe:
    family: str
    representation: str
    training: str = "mean"
    params: tuple = ()
    top_pairs: int = 8
    seed: int = 0


def config_from_bounds(factor_bounds, **settings):
    """Configure any number of continuous inputs from an ordered name -> bounds mapping.

    Counts of alternatives and repetitions are discovered from the input rows.
    Bounds are physical/domain limits, not inferred from held-out observations.
    """
    return Config(factors=tuple(factor_bounds), bounds=tuple(tuple(b) for b in factor_bounds.values()), **settings)
