from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TopoNaderDataConfig:
    input_key: str = "inputs"
    output_key: str = "outputs"
    chronological_input_key: str = "inputsX"
    chronological_output_key: str = "outputsX"
    num_bars: int = 4
    parameters_per_bar: int = 4
    validation_fraction: float = 0.2
    split_seed: int | None = None
    output_log10: bool = True

    @property
    def input_dim(self) -> int:
        return int(self.num_bars * self.parameters_per_bar)

    @property
    def design_shape(self) -> tuple[int, int]:
        return int(self.num_bars), int(self.parameters_per_bar)


@dataclass
class TopoNaderTrainingConfig:
    epochs: int = 300
    quick_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 40
    seed: int | None = None
    device: str = "auto"


@dataclass
class TopoNaderUncertaintyConfig:
    method: str = "mc_dropout"
    mc_samples: int = 20
    kappa: float = 1.5


@dataclass
class TopoNaderCoverageConfig:
    upper_bound_enabled: bool = True
    upper_bound: float = 0.30
    upper_tolerance: float = 0.02
    lower_bound_enabled: bool = False
    lower_bound: float | None = None


@dataclass
class TopoNaderEliteLocalSearchConfig:
    enabled: bool = False
    elite_count: int = 5
    particle_weights: tuple[float, ...] = (0.40, 0.20, 0.16, 0.12, 0.12)
    selected_weights: tuple[float, ...] = (0.40, 0.20, 0.20, 0.10, 0.10)
    small_position_radius: float = 20.0
    small_angle_radius: float = 5.0
    small_width_radius: float = 10.0
    medium_position_radius: float = 40.0
    medium_angle_radius: float = 10.0
    medium_width_radius: float = 20.0
    single_parameter_fraction: float = 0.60
    single_bar_two_parameter_fraction: float = 0.25
    candidates_per_elite: int = 100


@dataclass
class TopoNaderOptimizationConfig:
    particles: int = 300
    iterations: int = 500
    mutation_rate: float = 0.1
    stall_limit: int = 15
    candidate_pool_size: int = 100
    fe_batch_size: int = 20
    exploitation_count: int = 10
    exploration_count: int = 10
    coverage: TopoNaderCoverageConfig = field(default_factory=TopoNaderCoverageConfig)
    elite_local_search: TopoNaderEliteLocalSearchConfig = field(
        default_factory=TopoNaderEliteLocalSearchConfig
    )
    grid_nx: int = 55
    grid_ny: int = 90
    grid_step: float = 10.0
    grid_x0: float = -5.0
    grid_y0: float = 445.0
    rod_width_min: float = 1.0
    rod_width_max: float = 500.0

    def __post_init__(self) -> None:
        if isinstance(self.coverage, dict):
            self.coverage = TopoNaderCoverageConfig(**self.coverage)
        if isinstance(self.elite_local_search, dict):
            payload = dict(self.elite_local_search)
            for key in ("particle_weights", "selected_weights"):
                if key in payload:
                    payload[key] = tuple(float(value) for value in payload[key])
            self.elite_local_search = TopoNaderEliteLocalSearchConfig(**payload)


@dataclass
class TopoNaderTriggerConfig:
    enabled: bool = True
    patience: int = 3
    max_architecture_age: int = 10
    min_pairwise_accuracy: float = 0.60
    min_kendall_tau: float = 0.20
    max_topk_regret: float = 0.20
    min_feedback_samples: int = 4


@dataclass
class TopoNaderArchitectureConfig:
    max_parameters: int = 5_000_000
    candidate_count: int = 5
    fully_train_count: int = 3
    min_hidden_width: int = 16
    max_hidden_width: int = 2048
    max_depth: int = 8
    min_latent_dim: int = 2
    max_latent_dim: int = 512
    candidate_retry_limit: int = 3
    zero_cost_enabled: bool = True
    zero_cost_keep_count: int = 4


@dataclass
class TopoNaderMethodConfig:
    architecture_adaptation: bool = True
    architecture_review_mode: str = "event_triggered"
    uncertainty_sampling: bool = True
    diversity_sampling: bool = True


@dataclass
class TopoNaderReportingConfig:
    target_performance: float | None = None


@dataclass
class TopoNaderAgentConfig:
    enabled: bool = True
    model: str = "gpt-5.5"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 300
    max_retries: int = 3
    structured_output_retries: int = 2


@dataclass
class TopoNaderPathsConfig:
    runs_dir: str = "runs"
    experience_file: str = "state/architecture_experiences.jsonl"
    source_topology_dir: str = "matlab"
    comsol_model: str = "matlab/fuban_Beam.mph"


@dataclass
class TopoNaderConfig:
    experiment_name: str = "toponader-default"
    random_seed: int = 3
    data: TopoNaderDataConfig = field(default_factory=TopoNaderDataConfig)
    training: TopoNaderTrainingConfig = field(default_factory=TopoNaderTrainingConfig)
    uncertainty: TopoNaderUncertaintyConfig = field(default_factory=TopoNaderUncertaintyConfig)
    optimization: TopoNaderOptimizationConfig = field(default_factory=TopoNaderOptimizationConfig)
    trigger: TopoNaderTriggerConfig = field(default_factory=TopoNaderTriggerConfig)
    architecture: TopoNaderArchitectureConfig = field(default_factory=TopoNaderArchitectureConfig)
    method: TopoNaderMethodConfig = field(default_factory=TopoNaderMethodConfig)
    reporting: TopoNaderReportingConfig = field(default_factory=TopoNaderReportingConfig)
    agents: TopoNaderAgentConfig = field(default_factory=TopoNaderAgentConfig)
    paths: TopoNaderPathsConfig = field(default_factory=TopoNaderPathsConfig)

    def validate(self) -> None:
        name = self.experiment_name
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        invalid_character = any(ord(character) < 32 or character in '<>:"/\\|?*' for character in name)
        if (
            not name
            or len(name) > 128
            or name in {".", ".."}
            or name.endswith((" ", "."))
            or invalid_character
            or name.split(".", 1)[0].upper() in reserved
        ):
            raise ValueError("experiment_name must be a single path-safe folder name of at most 128 characters")
        if self.data.num_bars < 1:
            raise ValueError("num_bars must be positive")
        if self.data.parameters_per_bar != 4:
            raise ValueError("TopoNader currently requires 4 parameters per bar: xc, yc, theta and width")
        if not 0.0 < self.data.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.optimization.exploitation_count + self.optimization.exploration_count != self.optimization.fe_batch_size:
            raise ValueError("exploitation_count + exploration_count must equal fe_batch_size")
        elite = self.optimization.elite_local_search
        if elite.enabled:
            if elite.elite_count < 1:
                raise ValueError("elite_local_search elite_count must be positive")
            if self.optimization.particles < elite.elite_count:
                raise ValueError("elite_local_search requires at least one particle per elite")
            if len(elite.particle_weights) != elite.elite_count or any(
                weight <= 0 for weight in elite.particle_weights
            ):
                raise ValueError("elite_local_search particle_weights must contain one positive value per elite")
            if len(elite.selected_weights) != elite.elite_count or any(
                weight < 0 for weight in elite.selected_weights
            ) or sum(elite.selected_weights) <= 0:
                raise ValueError("elite_local_search selected_weights must contain one non-negative value per elite")
            radii = (
                elite.small_position_radius,
                elite.small_angle_radius,
                elite.small_width_radius,
                elite.medium_position_radius,
                elite.medium_angle_radius,
                elite.medium_width_radius,
            )
            if any(radius <= 0 for radius in radii):
                raise ValueError("elite_local_search radii must be positive")
            if not 0.0 <= elite.single_parameter_fraction <= 1.0:
                raise ValueError("elite_local_search single_parameter_fraction must be in [0, 1]")
            if not 0.0 <= elite.single_bar_two_parameter_fraction <= 1.0:
                raise ValueError("elite_local_search single_bar_two_parameter_fraction must be in [0, 1]")
            if elite.single_parameter_fraction + elite.single_bar_two_parameter_fraction > 1.0:
                raise ValueError("elite_local_search sparse initialization fractions exceed 1")
            if elite.candidates_per_elite < 1:
                raise ValueError("elite_local_search candidates_per_elite must be positive")
        if self.uncertainty.method != "mc_dropout":
            raise ValueError("only mc_dropout is implemented in this version")
        if self.architecture.candidate_count < 1 or self.architecture.fully_train_count < 1:
            raise ValueError("architecture candidate counts must be positive")
        if self.architecture.candidate_retry_limit < 1:
            raise ValueError("candidate_retry_limit must be positive")
        if self.architecture.zero_cost_keep_count < 1:
            raise ValueError("zero_cost_keep_count must be positive")
        if self.agents.max_retries < 1:
            raise ValueError("agents max_retries must be positive")
        if self.agents.structured_output_retries < 0:
            raise ValueError("agents structured_output_retries must be non-negative")
        if not 1 <= self.architecture.min_latent_dim <= self.architecture.max_latent_dim:
            raise ValueError("latent dimension bounds are invalid")
        if self.method.architecture_review_mode not in {"event_triggered", "every_round"}:
            raise ValueError("architecture_review_mode must be event_triggered or every_round")
        coverage = self.optimization.coverage
        if coverage.upper_bound_enabled:
            if not 0.0 <= coverage.upper_bound <= 1.0:
                raise ValueError("coverage upper_bound must be in [0, 1]")
            if coverage.upper_tolerance < 0.0:
                raise ValueError("coverage upper_tolerance must be non-negative")
        if coverage.lower_bound_enabled:
            if coverage.lower_bound is None or not 0.0 <= coverage.lower_bound <= 1.0:
                raise ValueError("enabled coverage lower_bound must be in [0, 1]")
            if coverage.upper_bound_enabled and coverage.lower_bound > coverage.upper_bound + coverage.upper_tolerance:
                raise ValueError("coverage lower_bound exceeds the allowed upper bound")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def seed_for(self, namespace: str) -> int:
        """Derive a stable independent 32-bit seed from the experiment seed."""
        override = None
        if namespace == "data_split":
            override = self.data.split_seed
        elif namespace == "training":
            override = self.training.seed
        if override is not None:
            return int(override)
        digest = hashlib.sha256(f"{int(self.random_seed)}:{namespace}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big", signed=False)

    def seed_manifest(self) -> dict[str, int]:
        namespaces = (
            "data_split",
            "training",
            "pso_mean",
            "pso_ucb",
            "mc_dropout",
            "zero_cost",
            "matlab_inputs",
        )
        return {name: self.seed_for(name) for name in namespaces}

    def default_run_dir(self, project_root: str | Path) -> Path:
        return Path(project_root).resolve() / self.paths.runs_dir / self.experiment_name


_SECTIONS = {
    "data": TopoNaderDataConfig,
    "training": TopoNaderTrainingConfig,
    "uncertainty": TopoNaderUncertaintyConfig,
    "optimization": TopoNaderOptimizationConfig,
    "trigger": TopoNaderTriggerConfig,
    "architecture": TopoNaderArchitectureConfig,
    "method": TopoNaderMethodConfig,
    "reporting": TopoNaderReportingConfig,
    "agents": TopoNaderAgentConfig,
    "paths": TopoNaderPathsConfig,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_payload(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    visited = set() if visited is None else visited
    if path in visited:
        raise ValueError(f"cyclic config inheritance at {path}")
    visited.add(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    base = _load_payload((path.parent / parent).resolve(), visited)
    return _deep_merge(base, payload)


def load_config(path: str | Path) -> TopoNaderConfig:
    payload = _load_payload(Path(path))
    optimization = payload.get("optimization")
    if isinstance(optimization, dict):
        optimization = dict(optimization)
        legacy_target = optimization.pop("ratio_target", None)
        legacy_tolerance = optimization.pop("ratio_tolerance", None)
        if legacy_target is not None or legacy_tolerance is not None:
            coverage = dict(optimization.get("coverage", {}))
            if legacy_target is not None:
                coverage["upper_bound"] = legacy_target
            if legacy_tolerance is not None:
                coverage["upper_tolerance"] = legacy_tolerance
            optimization["coverage"] = coverage
        payload["optimization"] = optimization
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        kwargs[key] = _SECTIONS[key](**value) if key in _SECTIONS else value
    config = TopoNaderConfig(**kwargs)
    config.validate()
    return config


def save_config(config: TopoNaderConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
