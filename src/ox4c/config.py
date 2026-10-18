from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OptimizationConfig:
    algorithm: str = "agraspqfs"
    preset: str | None = "fast"
    min_features: int = 3
    max_features: int | None = 30
    constructive_iterations: int | None = None
    local_iterations: int | None = None
    priority_queue_size: int | None = None
    rcl_size: int | None = None
    alpha: float = 0.7
    update_interval: int = 10
    ranking_method: str = "mutual_info"
    validation_fraction: float | None = None
    cv: int | None = None
    max_evaluations: int | None = 30
    early_stopping_rounds: int | None = None
    time_budget: float | None = 180.0
    evaluation_sample_size: float = 5000
    max_validation_mcc_loss: float = 0.01
    min_feature_reduction_fraction: float = 0.20


@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    policy: str = "alerts_and_collaborative"
    top_k: int = 10


@dataclass
class ComplianceConfig:
    enabled: bool = False
    enforce: bool = True
    policy_id: str = "ox4c-technical-conformance-v1"
    required_provenance_fields: list[str] = field(default_factory=lambda: [
        "source_file", "sha256", "target", "split_policy", "split_sizes",
    ])
    require_model_version_evidence: bool = True
    require_audit_record: bool = True
    require_explanation_capability: bool = True
    feature_allowlist: list[str] = field(default_factory=list)
    feature_denylist: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    algorithm: str = "random_forest"
    n_estimators: int = 160
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    random_state: int = 42
    inference_jobs: int = 1
    hgb_max_iter: int = 160
    hgb_learning_rate: float = 0.08
    hgb_max_leaf_nodes: int = 31
    hgb_l2_regularization: float = 1.0
    logistic_max_iter: int = 1000
    compact_max_depth: int = 10
    extra_trees_estimators: int = 200
    attack_sensitive_weight: float = 2.0


@dataclass
class CollaborationConfig:
    enabled: bool = True
    primary_algorithm: str = "random_forest"
    counselor_algorithms: list[str] = field(default_factory=lambda: [
        "hist_gradient_boosting", "compact_tree",
    ])
    consensus: str = "calibration_selected_vote"
    minimum_calibration_mcc_gain: float = 0.0


@dataclass
class LearningConfig:
    enabled: bool = True
    feedback_delay_events: int = 50
    batch_size: int = 200
    max_promotions: int = 10


@dataclass
class CompactModelConfig:
    enabled: bool = True
    algorithm: str = "compact_tree"
    benchmark_samples: int = 512


@dataclass
class OX4CConfig:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    collaboration: CollaborationConfig = field(default_factory=CollaborationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    compact_model: CompactModelConfig = field(default_factory=CompactModelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "artifacts/run"

    def to_dict(self) -> dict:
        return asdict(self)
