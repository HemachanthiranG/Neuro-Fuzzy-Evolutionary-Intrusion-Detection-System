"""Configuration loader and management for the NSL-KDD Feature Selection project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PathsConfig:
    raw_train_path: Path
    raw_test_path: Path
    processed_dir: Path
    models_dir: Path
    results_dir: Path
    plots_dir: Path
    confusion_matrices_dir: Path


@dataclass
class DataConfig:
    column_names: List[str]
    categorical_features: List[str]
    target_column: str
    difficulty_column: str
    val_split_ratio: float
    scaler_type: str
    classes: List[str]
    attack_mapping: Dict[str, List[str]]


@dataclass
class ANNConfig:
    hidden_layers: List[int]
    dropouts: List[float]
    activations: List[str]
    learning_rate: float
    batch_size: int
    epochs: int
    use_batch_norm: bool
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_lr: float
    use_class_weights: bool


@dataclass
class GAConfig:
    chromosome_length: int
    population_size: int
    generations: int
    tournament_size: int
    crossover_rate: float
    mutation_rate: float
    elitism_count: int
    alpha_penalty: float
    early_stopping_generations: int
    eval_subsample_ratio: float
    fast_ann_epochs: int
    fast_ann_batch_size: int
    beta_penalty: float = 0.3
    use_fuzzy_controller: bool = False
    seed_informed_chromosomes: bool = True


@dataclass
class FuzzyInputConfig:
    num_terms: int = 3
    input_mode: str = "concat"  # "concat" or "membership_only"
    fcm_m: float = 2.0
    fcm_max_iter: int = 100
    fcm_error: float = 0.0001
    min_variance: float = 0.00001


@dataclass
class FuzzyAdaptiveGAConfig:
    enabled: bool = True
    min_mutation_rate: float = 0.005
    max_mutation_rate: float = 0.10
    min_crossover_rate: float = 0.60
    max_crossover_rate: float = 0.95
    beta_penalty: float = 0.3
    seed_informed_chromosomes: bool = True


@dataclass
class FuzzyDecisionConfig:
    enabled: bool = True
    beta_fpr_penalty: float = 0.30
    workload_penalty: float = 0.10
    false_alarm_penalty: float = 0.30
    suspicious_band: List[float] = field(default_factory=lambda: [0.35, 0.65])
    alert_levels: List[str] = field(default_factory=lambda: ["Normal", "Suspicious", "Attack"])
    rules: List[str] = field(default_factory=list)


@dataclass
class FuzzyConfig:
    input_layer: FuzzyInputConfig
    adaptive_ga: FuzzyAdaptiveGAConfig
    decision_layer: FuzzyDecisionConfig


@dataclass
class AppConfig:
    project_name: str
    paths: PathsConfig
    data: DataConfig
    ann: ANNConfig
    ga: GAConfig
    fuzzy: FuzzyConfig
    seeds: List[int]
    train_subsample: float = 1.0
    mode: str = "full"


def load_config(config_path: str = "config.yaml", mode: str = "fast", seeds: Optional[List[int]] = None) -> AppConfig:
    """Load configuration from YAML and apply execution mode overrides."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        # Fall back to project root (parent directory of src/)
        fallback_path = Path(__file__).resolve().parent.parent / config_path
        if fallback_path.exists():
            resolved_path = fallback_path

    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(__file__).resolve().parent.parent

    def _resolve(p_str: str) -> Path:
        p = Path(p_str)
        return p if p.is_absolute() else (project_root / p)

    paths_data = cfg.get("paths", {})
    paths = PathsConfig(
        raw_train_path=_resolve(paths_data.get("raw_train_path", "data/raw/KDDTrain+.txt")),
        raw_test_path=_resolve(paths_data.get("raw_test_path", "data/raw/KDDTest+.txt")),
        processed_dir=_resolve(paths_data.get("processed_dir", "data/processed")),
        models_dir=_resolve(paths_data.get("models_dir", "models")),
        results_dir=_resolve(paths_data.get("results_dir", "results")),
        plots_dir=_resolve(paths_data.get("plots_dir", "results/plots")),
        confusion_matrices_dir=_resolve(paths_data.get("confusion_matrices_dir", "results/confusion_matrices")),
    )

    data_dict = cfg.get("data", {})
    data = DataConfig(
        column_names=data_dict.get("column_names", []),
        categorical_features=data_dict.get("categorical_features", []),
        target_column=data_dict.get("target_column", "label"),
        difficulty_column=data_dict.get("difficulty_column", "difficulty"),
        val_split_ratio=data_dict.get("val_split_ratio", 0.2),
        scaler_type=data_dict.get("scaler_type", "standard"),
        classes=data_dict.get("classes", []),
        attack_mapping=data_dict.get("attack_mapping", {}),
    )

    ann_dict = dict(cfg.get("ann", {}))
    ga_dict = dict(cfg.get("ga", {}))

    # Apply execution mode overrides
    exec_modes = cfg.get("execution_modes", {})
    mode_cfg = exec_modes.get(mode, {})
    default_seeds = mode_cfg.get("seeds", [42])
    train_subsample = mode_cfg.get("train_subsample", 1.0)

    if "ga" in mode_cfg:
        ga_dict.update(mode_cfg["ga"])
    if "ann" in mode_cfg:
        ann_dict.update(mode_cfg["ann"])

    ann = ANNConfig(
        hidden_layers=ann_dict.get("hidden_layers", [128, 64]),
        dropouts=ann_dict.get("dropouts", [0.3, 0.2]),
        activations=ann_dict.get("activations", ["relu", "relu"]),
        learning_rate=float(ann_dict.get("learning_rate", 0.001)),
        batch_size=int(ann_dict.get("batch_size", 128)),
        epochs=int(ann_dict.get("epochs", 30)),
        use_batch_norm=bool(ann_dict.get("use_batch_norm", False)),
        early_stopping_patience=int(ann_dict.get("early_stopping_patience", 5)),
        reduce_lr_patience=int(ann_dict.get("reduce_lr_patience", 3)),
        reduce_lr_factor=float(ann_dict.get("reduce_lr_factor", 0.5)),
        min_lr=float(ann_dict.get("min_lr", 0.00001)),
        use_class_weights=bool(ann_dict.get("use_class_weights", True)),
    )

    ga = GAConfig(
        chromosome_length=int(ga_dict.get("chromosome_length", 41)),
        population_size=int(ga_dict.get("population_size", 20)),
        generations=int(ga_dict.get("generations", 15)),
        tournament_size=int(ga_dict.get("tournament_size", 3)),
        crossover_rate=float(ga_dict.get("crossover_rate", 0.8)),
        mutation_rate=float(ga_dict.get("mutation_rate", 1.0 / 41.0)),
        elitism_count=int(ga_dict.get("elitism_count", 2)),
        alpha_penalty=float(ga_dict.get("alpha_penalty", 0.02)),
        beta_penalty=float(ga_dict.get("beta_penalty", 0.05)),
        early_stopping_generations=int(ga_dict.get("early_stopping_generations", 4)),
        eval_subsample_ratio=float(ga_dict.get("eval_subsample_ratio", 0.25)),
        fast_ann_epochs=int(ga_dict.get("fast_ann_epochs", 4)),
        fast_ann_batch_size=int(ga_dict.get("fast_ann_batch_size", 256)),
        use_fuzzy_controller=bool(ga_dict.get("use_fuzzy_controller", False)),
        seed_informed_chromosomes=bool(ga_dict.get("seed_informed_chromosomes", False)),
    )

    # Fuzzy logic configuration
    fuzzy_dict = cfg.get("fuzzy", {})
    in_layer_dict = fuzzy_dict.get("input_layer", {})
    f_input = FuzzyInputConfig(
        num_terms=int(in_layer_dict.get("num_terms", 3)),
        input_mode=str(in_layer_dict.get("input_mode", "concat")),
        fcm_m=float(in_layer_dict.get("fcm_m", 2.0)),
        fcm_max_iter=int(in_layer_dict.get("fcm_max_iter", 100)),
        fcm_error=float(in_layer_dict.get("fcm_error", 0.0001)),
        min_variance=float(in_layer_dict.get("min_variance", 0.00001)),
    )

    ad_ga_dict = fuzzy_dict.get("adaptive_ga", {})
    f_ad_ga = FuzzyAdaptiveGAConfig(
        enabled=bool(ad_ga_dict.get("enabled", True)),
        min_mutation_rate=float(ad_ga_dict.get("min_mutation_rate", 0.005)),
        max_mutation_rate=float(ad_ga_dict.get("max_mutation_rate", 0.10)),
        min_crossover_rate=float(ad_ga_dict.get("min_crossover_rate", 0.60)),
        max_crossover_rate=float(ad_ga_dict.get("max_crossover_rate", 0.95)),
        beta_penalty=float(ad_ga_dict.get("beta_penalty", 0.3)),
        seed_informed_chromosomes=bool(ad_ga_dict.get("seed_informed_chromosomes", True)),
    )

    dec_dict = fuzzy_dict.get("decision_layer", {})
    f_decision = FuzzyDecisionConfig(
        enabled=bool(dec_dict.get("enabled", True)),
        beta_fpr_penalty=float(dec_dict.get("beta_fpr_penalty", 0.30)),
        suspicious_band=list(dec_dict.get("suspicious_band", [0.35, 0.65])),
        alert_levels=list(dec_dict.get("alert_levels", ["Normal", "Suspicious", "Attack"])),
        rules=list(dec_dict.get("rules", [])),
    )

    fuzzy = FuzzyConfig(
        input_layer=f_input,
        adaptive_ga=f_ad_ga,
        decision_layer=f_decision,
    )

    final_seeds = seeds if seeds is not None else default_seeds

    return AppConfig(
        project_name=cfg.get("project_name", "NSL-KDD Feature Selection"),
        paths=paths,
        data=data,
        ann=ann,
        ga=ga,
        fuzzy=fuzzy,
        seeds=final_seeds,
        train_subsample=train_subsample,
        mode=mode,
    )
