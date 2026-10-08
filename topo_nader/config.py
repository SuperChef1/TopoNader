from __future__ import annotations

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
    split_seed: int = 3
    output_log10: bool = True


@dataclass
class TopoNaderTrainingConfig:
    epochs: int = 300
    quick_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 40
    seed: int = 3
    device: str = "auto"


@dataclass
class TopoNaderUncertaintyConfig:
    method: str = "mc_dropout"
    mc_samples: int = 20
    kappa: float = 1.5


@dataclass
class TopoNaderOptimizationConfig:
    particles: int = 300
    iterations: int = 500
    mutation_rate: float = 0.1
    stall_limit: int = 15
    candidate_pool_size: int = 100
    fe_batch_size: int = 10
    exploitation_count: int = 5
    exploration_count: int = 5
    ratio_target: float = 0.30
    ratio_tolerance: float = 0.02
    grid_nx: int = 55
    grid_ny: int = 90
    grid_step: float = 10.0
    grid_x0: float = -5.0
    grid_y0: float = 445.0
    rod_width_min: float = 1.0
    rod_width_max: float = 500.0


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


@dataclass
class TopoNaderPathsConfig:
    runs_dir: str = "runs"
    experience_file: str = "state/architecture_experiences.jsonl"
    source_topology_dir: str = "matlab"
    comsol_model: str = "matlab/fuban_Beam.mph"


@dataclass
class TopoNaderConfig:
    experiment_name: str = "toponader-default"
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
        if self.data.num_bars * self.data.parameters_per_bar != 16:
            raise ValueError("TopoNader currently requires 4 bars x 4 parameters = 16 inputs")
        if not 0.0 < self.data.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.optimization.exploitation_count + self.optimization.exploration_count != self.optimization.fe_batch_size:
            raise ValueError("exploitation_count + exploration_count must equal fe_batch_size")
        if self.uncertainty.method != "mc_dropout":
            raise ValueError("only mc_dropout is implemented in this version")
        if self.architecture.candidate_count < 1 or self.architecture.fully_train_count < 1:
            raise ValueError("architecture candidate counts must be positive")
        if self.architecture.candidate_retry_limit < 1:
            raise ValueError("candidate_retry_limit must be positive")
        if self.architecture.zero_cost_keep_count < 1:
            raise ValueError("zero_cost_keep_count must be positive")
        if self.method.architecture_review_mode not in {"event_triggered", "every_round"}:
            raise ValueError("architecture_review_mode must be event_triggered or every_round")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
