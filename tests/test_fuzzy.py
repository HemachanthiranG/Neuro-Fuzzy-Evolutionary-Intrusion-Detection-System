"""Unit and integration tests for Fuzzy Logic components (Phase F1)."""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import FuzzyInputConfig, load_config
from src.fuzzy import FuzzyCMeans1D, FuzzyInputLayer
from src.preprocessing import process_dataset_and_generate_eda
from src.utils import load_artifact, load_json


def test_fuzzy_cmeans_normal():
    """Verify 1D FCM converges on normal data and produces valid memberships in [0, 1]."""
    rng = np.random.RandomState(42)
    data = rng.normal(loc=10.0, scale=2.0, size=500)

    fcm = FuzzyCMeans1D(n_clusters=3, m=2.0)
    fcm.fit(data)

    assert fcm.is_fitted
    assert len(fcm.centers_) == 3
    assert len(fcm.sigmas_) == 3
    # Centers should be sorted in ascending order
    assert np.all(np.diff(fcm.centers_) > 0)
    assert np.all(fcm.sigmas_ > 0)

    m = fcm.compute_memberships(data)
    assert m.shape == (500, 3)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)
    assert not np.isnan(m).any()
    assert not np.isinf(m).any()


def test_fuzzy_cmeans_edge_cases():
    """Verify FCM handles constant (zero-variance) and binary features without NaNs or zero-division."""
    # Constant data
    constant_data = np.full(200, 5.0)
    fcm_const = FuzzyCMeans1D(n_clusters=3)
    fcm_const.fit(constant_data)
    assert fcm_const.is_fitted
    m_const = fcm_const.compute_memberships(constant_data)
    assert m_const.shape == (200, 3)
    assert not np.isnan(m_const).any()
    assert np.all(m_const >= 0.0) and np.all(m_const <= 1.0)

    # Binary data (e.g. land, logged_in)
    binary_data = np.array([0, 1] * 100, dtype=float)
    fcm_bin = FuzzyCMeans1D(n_clusters=3)
    fcm_bin.fit(binary_data)
    assert fcm_bin.is_fitted
    m_bin = fcm_bin.compute_memberships(binary_data)
    assert m_bin.shape == (200, 3)
    assert not np.isnan(m_bin).any()
    assert np.all(m_bin >= 0.0) and np.all(m_bin <= 1.0)


def test_fuzzy_input_layer_transform():
    """Verify FuzzyInputLayer fits across multiple features and transforms correctly."""
    rng = np.random.RandomState(42)
    X = rng.rand(100, 4)
    feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]

    cfg = FuzzyInputConfig(num_terms=3, input_mode="concat")
    layer = FuzzyInputLayer(config=cfg)
    layer.fit(X, feature_names)

    assert layer.is_fitted
    m = layer.transform(X, feature_names)
    # 4 features * 3 terms = 12 columns
    assert m.shape == (100, 12)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)
    assert not np.isnan(m).any()


def test_preprocessor_fuzzy_integration():
    """Verify NSLKDDPreprocessor fits fuzzy layer on training split and saves parameters."""
    cfg = load_config(mode="smoke")
    datasets, preprocessor = process_dataset_and_generate_eda(config=cfg, seed=42)

    assert preprocessor.fuzzy_input_layer is not None
    assert preprocessor.fuzzy_input_layer.is_fitted
    assert len(preprocessor.fuzzy_input_layer.numeric_feature_names) == 38

    # Verify fuzzy membership parameters JSON was generated
    params_path = cfg.paths.results_dir / "fuzzy_membership_params.json"
    assert params_path.exists(), f"Expected {params_path} to exist."
    params_dict = load_json(params_path)
    assert len(params_dict) == 38

    # Test transform_neuro_fuzzy with a test chromosome
    chrom = np.zeros(41, dtype=int)
    chrom[0] = 1  # duration (numeric)
    chrom[1] = 1  # protocol_type (categorical, 3 cols)

    X_train = datasets["X_train"]
    # Concat mode: 1 numeric + 3 cat + (1 numeric * 3 fuzzy terms) = 7 columns
    nf_concat = preprocessor.transform_neuro_fuzzy(X_train[:50], chrom, input_mode="concat")
    assert nf_concat.shape == (50, 7)
    assert not np.isnan(nf_concat).any()

    # Membership-only mode: 3 cat cols + (1 numeric * 3 fuzzy terms) = 6 columns
    nf_mem = preprocessor.transform_neuro_fuzzy(X_train[:50], chrom, input_mode="membership_only")
    assert nf_mem.shape == (50, 6)
    assert not np.isnan(nf_mem).any()

    # Test reloaded preprocessor from disk
    saved_prep = load_artifact(cfg.paths.models_dir / "preprocessor.joblib")
    assert saved_prep.fuzzy_input_layer is not None
    nf_reloaded = saved_prep.transform_neuro_fuzzy(X_train[:50], chrom, input_mode="concat")
    assert np.array_equal(nf_concat, nf_reloaded)


def test_fuzzy_ga_controller_bounds():
    """Verify FuzzyGAController outputs strictly adhere to bounds across all input domains."""
    from src.fuzzy import FuzzyGAController

    controller = FuzzyGAController(
        min_mutation_rate=0.005,
        max_mutation_rate=0.10,
        min_crossover_rate=0.60,
        max_crossover_rate=0.95,
        max_stagnation_patience=6,
    )

    # Grid search across all diversity and stagnation levels
    for div in np.linspace(0.0, 1.0, 11):
        for stag in np.linspace(0.0, 1.0, 11):
            pm, pc, debug = controller.evaluate(diversity=float(div), stagnation=float(stag))
            assert 0.005 <= pm <= 0.10, f"Mutation rate {pm} out of bounds for div={div}, stag={stag}"
            assert 0.60 <= pc <= 0.95, f"Crossover rate {pc} out of bounds for div={div}, stag={stag}"
            assert not np.isnan(pm) and not np.isnan(pc)

    # Specific rule logic check:
    # Low diversity & high stagnation -> High mutation (escapes local optima)
    pm_stuck, pc_stuck, _ = controller.evaluate(diversity=0.0, stagnation=1.0)
    # High diversity & low stagnation -> Low mutation, high crossover (exploits good search)
    pm_thriving, pc_thriving, _ = controller.evaluate(diversity=1.0, stagnation=0.0)

    assert pm_stuck > pm_thriving, f"Expected pm_stuck ({pm_stuck}) > pm_thriving ({pm_thriving})"
    assert pc_thriving > pc_stuck, f"Expected pc_thriving ({pc_thriving}) > pc_stuck ({pc_stuck})"

    rules = controller.export_rules_dict()
    assert len(rules) == 9
    assert all("antecedent" in r and "consequent" in r for r in rules)


def test_population_diversity_metric():
    """Verify vectorized pairwise Hamming diversity calculation."""
    from src.fuzzy import FuzzyGAController

    # All identical chromosomes -> diversity should be 0.0
    identical_pop = [np.ones(41, dtype=int) for _ in range(10)]
    div_zero = FuzzyGAController.compute_population_diversity(identical_pop)
    assert div_zero == 0.0

    # Half ones, half zeros -> high diversity
    diverse_pop = [
        np.array([1 if i % 2 == 0 else 0 for i in range(41)], dtype=int),
        np.array([0 if i % 2 == 0 else 1 for i in range(41)], dtype=int),
    ]
    div_high = FuzzyGAController.compute_population_diversity(diverse_pop)
    assert div_high > 0.9


def test_ga_repair_and_controller_integration():
    """Verify GA repair guarantee and GeneticFeatureSelector initialization with fuzzy controller."""
    from src.ga import GeneticFeatureSelector

    cfg = load_config(mode="smoke")
    datasets, preprocessor = process_dataset_and_generate_eda(config=cfg, seed=42)

    ga = GeneticFeatureSelector(
        ga_config=cfg.ga,
        ann_config=cfg.ann,
        preprocessor=preprocessor,
        class_weights=datasets["class_weights"],
        seed=42,
        use_fuzzy_controller=True,
        seed_informed_chromosomes=True,
    )

    # Test repair on empty chromosome
    empty_chrom = np.zeros(41, dtype=int)
    repaired = ga.repair_chromosome(empty_chrom)
    assert np.sum(repaired) >= 1, "Repaired chromosome must have at least one active feature."

    # Test population initialization with informed seeding
    pop = ga.initialize_population(datasets["X_train"][:100], datasets["y_train"][:100])
    assert len(pop) == cfg.ga.population_size
    for ind in pop:
        assert ind.shape == (41,)
        assert np.sum(ind) >= 1


def test_ga_entrypoint_never_receives_test_data():
    """Assert GA entry points and evaluation methods receive only train and val data, never test data."""
    import inspect
    from src.ga import GeneticFeatureSelector

    # 1. Verify signatures of GA core methods contain NO test parameters
    evolve_params = list(inspect.signature(GeneticFeatureSelector.evolve).parameters.keys())
    assert not any("test" in p.lower() for p in evolve_params), (
        f"Unexpected test parameter in evolve: {evolve_params}"
    )

    eval_params = list(inspect.signature(GeneticFeatureSelector.evaluate_chromosome).parameters.keys())
    assert not any("test" in p.lower() for p in eval_params), (
        f"Unexpected test parameter in evaluate_chromosome: {eval_params}"
    )

    # 2. Verify evolve executes successfully when test set is explicitly deleted/poisoned
    cfg = load_config(mode="smoke")
    datasets, preprocessor = process_dataset_and_generate_eda(config=cfg, seed=42)

    # Corrupt/poison test split to prove evolve NEVER touches it
    datasets["X_test"] = None
    datasets["y_test"] = None

    ga = GeneticFeatureSelector(
        ga_config=cfg.ga,
        ann_config=cfg.ann,
        preprocessor=preprocessor,
        class_weights=datasets["class_weights"],
        seed=42,
    )

    best_chrom, best_fit, history_df = ga.evolve(
        X_train=datasets["X_train"][:200],
        y_train=datasets["y_train"][:200],
        X_val=datasets["X_val"][:100],
        y_val=datasets["y_val"][:100],
    )
    assert len(best_chrom) == 41
    assert not history_df.empty


def test_fuzzy_decision_layer_outputs():
    """Verify FuzzyDecisionLayer produces valid severity scores, alert levels, and predictions."""
    from src.fuzzy import FuzzyDecisionLayer

    layer = FuzzyDecisionLayer(suspicious_band=[0.35, 0.65])

    # 1. Obvious Normal record
    prob_normal = np.array([[0.95, 0.02, 0.01, 0.01, 0.01]])
    res_normal = layer.predict(prob_normal)
    assert res_normal["severity"][0] < 0.35, f"Expected low severity for Normal, got {res_normal['severity'][0]}"
    assert res_normal["alert_indices"][0] == 0  # Normal
    assert res_normal["alert_names"][0] == "Normal"
    assert res_normal["hard_preds"][0] == 0
    assert res_normal["fuzzy_preds"][0] == 0

    # 2. Obvious Attack record (DoS)
    prob_attack = np.array([[0.01, 0.95, 0.02, 0.01, 0.01]])
    res_attack = layer.predict(prob_attack)
    assert res_attack["severity"][0] > 0.65, f"Expected high severity for Attack, got {res_attack['severity'][0]}"
    assert res_attack["alert_indices"][0] == 2  # Attack
    assert res_attack["alert_names"][0] == "Attack"
    assert res_attack["hard_preds"][0] == 1
    assert res_attack["fuzzy_preds"][0] == 1

    # 3. Highly ambiguous record (uniform distribution -> high entropy)
    prob_uniform = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])
    res_uniform = layer.predict(prob_uniform)
    assert 0.35 <= res_uniform["severity"][0] <= 0.65, f"Expected suspicious severity for uniform, got {res_uniform['severity'][0]}"
    assert res_uniform["alert_indices"][0] == 1  # Suspicious
    assert res_uniform["alert_names"][0] == "Suspicious"

    # 4. Batch properties check
    rng = np.random.RandomState(42)
    raw_probs = rng.uniform(0.01, 1.0, size=(100, 5))
    batch_probs = raw_probs / np.sum(raw_probs, axis=1, keepdims=True)

    batch_res = layer.predict(batch_probs)
    assert batch_res["severity"].shape == (100,)
    assert np.all(batch_res["severity"] >= 0.0) and np.all(batch_res["severity"] <= 1.0)
    assert not np.isnan(batch_res["severity"]).any()
    assert np.all(np.isin(batch_res["alert_indices"], [0, 1, 2]))
    assert np.all(np.isin(batch_res["fuzzy_preds"], [0, 1, 2, 3, 4]))


def test_fuzzy_decision_layer_tuning_leakage_prevention():
    """Verify FuzzyDecisionLayer tunes strictly on validation split and forbids test arrays."""
    import inspect
    from src.fuzzy import FuzzyDecisionLayer

    # 1. Signature introspection: MUST NOT have 'test' parameters
    tune_params = list(inspect.signature(FuzzyDecisionLayer.tune_on_validation).parameters.keys())
    assert not any("test" in p.lower() for p in tune_params), (
        f"Security violation: 'test' found in tune_on_validation parameters: {tune_params}"
    )

    # 2. Execution check on simulated validation predictions
    rng = np.random.RandomState(42)
    val_probs = rng.uniform(0.01, 1.0, size=(200, 5))
    val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
    y_val = rng.choice([0, 1, 2, 3, 4], size=200, p=[0.5, 0.3, 0.1, 0.07, 0.03])

    layer = FuzzyDecisionLayer()
    stats = layer.tune_on_validation(val_probs, y_val, beta_fpr_penalty=0.3)

    assert layer.is_tuned
    assert "validation_baseline" in stats
    assert "validation_fuzzy_adjusted" in stats
    assert "tuned_suspicious_band" in stats
    assert "suspicious_band_validation_stats" in stats
    assert len(layer.suspicious_band) == 2
    assert layer.suspicious_band[0] < layer.suspicious_band[1]


def test_export_all_fuzzy_rules(tmp_path):
    """Verify that export_all_fuzzy_rules produces valid specification of both FLC and Decision Layer rules."""
    from src.fuzzy import export_all_fuzzy_rules
    from src.utils import load_json

    out_file = tmp_path / "test_rules.json"
    rules_dict = export_all_fuzzy_rules(out_file)

    assert out_file.exists()
    loaded = load_json(out_file)
    assert "mamdani_ga_controller" in loaded
    assert "fuzzy_decision_layer" in loaded
    assert len(loaded["mamdani_ga_controller"]["rules"]) == 9
    assert len(loaded["fuzzy_decision_layer"]["rules"]) == 9


def test_win_loss_logic_rules():
    """Verify win/loss logic enforces n<3 INSUFFICIENT DATA, n=1 std exclusion, and <10% latency noise."""
    from src.run_experiments import generate_win_loss_table

    # Mock summary with identical Macro-F1 (tie) and latency within 10%
    mock_summary = {
        "Full_Model": {
            "test_macro_f1_mean": 0.6000,
            "test_macro_f1_std": 0.0,
            "inference_time_per_1000_ms_mean": 68.0,  # +4.6% vs 65.0 (<10%)
            "inference_time_per_1000_ms_std": 0.0,
            "num_features_original_mean": 15.0,
            "num_features_original_std": 0.0,
        },
        "Baseline": {
            "test_macro_f1_mean": 0.6001,
            "test_macro_f1_std": 0.0,
            "inference_time_per_1000_ms_mean": 65.0,
            "inference_time_per_1000_ms_std": 0.0,
            "num_features_original_mean": 41.0,
            "num_features_original_std": 0.0,
        },
    }

    # 1. With n_seeds = 1: Macro-F1 difference is 0.0001 (< 0.005) -> INSUFFICIENT DATA (n<3)
    # Latency diff is 4.6% (< 10%) -> WITHIN NOISE
    # Features diff is -26 -> WIN
    df_1, stats_1 = generate_win_loss_table(
        summary_dict=mock_summary,
        full_model="Full_Model",
        baseline_model="Baseline",
        n_seeds=1,
    )
    f1_row_1 = df_1[df_1["Metric"] == "Macro-F1"].iloc[0]
    lat_row_1 = df_1[df_1["Metric"].str.contains("Latency")].iloc[0]
    feat_row_1 = df_1[df_1["Metric"] == "Original Features"].iloc[0]

    assert f1_row_1["Outcome"] == "INSUFFICIENT DATA (n<3)"
    assert lat_row_1["Outcome"] == "WITHIN NOISE"
    assert feat_row_1["Outcome"] == "WIN"
    # Verify no '±' in n=1 columns
    assert "±" not in f1_row_1["Full_Model"]
    assert "±" not in f1_row_1["Baseline"]
    assert stats_1["insufficient_data"] >= 1
    assert stats_1["within_noise"] == 1

    # 2. With n_seeds = 3: Macro-F1 diff < tol becomes TIE
    mock_summary_3 = {
        "Full_Model": {
            "test_macro_f1_mean": 0.6000,
            "test_macro_f1_std": 0.01,
            "inference_time_per_1000_ms_mean": 68.0,
            "inference_time_per_1000_ms_std": 1.0,
            "num_features_original_mean": 15.0,
            "num_features_original_std": 1.0,
        },
        "Baseline": {
            "test_macro_f1_mean": 0.6001,
            "test_macro_f1_std": 0.01,
            "inference_time_per_1000_ms_mean": 65.0,
            "inference_time_per_1000_ms_std": 1.0,
            "num_features_original_mean": 41.0,
            "num_features_original_std": 0.0,
        },
    }
    df_3, stats_3 = generate_win_loss_table(
        summary_dict=mock_summary_3,
        full_model="Full_Model",
        baseline_model="Baseline",
        n_seeds=3,
    )
    f1_row_3 = df_3[df_3["Metric"] == "Macro-F1"].iloc[0]
    assert f1_row_3["Outcome"] == "TIE"
    assert "±" in f1_row_3["Full_Model (Mean±Std)"]


def test_alert_metrics_and_model_isolation():
    """Verify alert-level metrics computation and strict isolation of 5-class predictions."""
    from src.fuzzy import FuzzyDecisionLayer

    layer = FuzzyDecisionLayer(suspicious_band=[0.35, 0.65])

    # 4 records:
    # 0: True Normal (0), alert 0 (Normal) -> correct normal, unflagged
    # 1: True Attack (1), alert 1 (Suspicious) -> flagged attack (recall), not alert 2
    # 2: True Attack (2), alert 2 (Attack) -> flagged attack, alert 2
    # 3: True Normal (0), alert 1 (Suspicious) -> flagged normal (false alarm)
    alert_indices = np.array([0, 1, 2, 1], dtype=np.int32)
    y_true = np.array([0, 1, 2, 0], dtype=np.int32)
    raw_attacks = np.array(["normal", "neptune", "ipsweep", "normal"])

    metrics = layer.compute_alert_metrics(alert_indices, y_true, raw_attacks=raw_attacks)

    # 2 true attacks: records 1 and 2. Both flagged (alert >= 1). Recall = 2/2 = 1.0
    assert metrics["attack_recall_flagged"] == 1.0

    # Records with alert == 2: record 2 (True Attack). Precision = 1/1 = 1.0
    assert metrics["attack_alert_precision"] == 1.0

    # 2 true normals: records 0 and 3. Record 3 is flagged (alert=1). False alarm rate = 1/2 = 0.5
    assert metrics["normal_flagged_rate"] == 0.5

    # Total flagged: records 1, 2, 3 -> 3/4 = 0.75
    assert metrics["analyst_workload_rate"] == 0.75

    # Per attack: neptune is flagged (1.0), ipsweep is flagged (1.0)
    assert metrics["per_attack_flagged_fraction"]["neptune"] == 1.0
    assert metrics["per_attack_flagged_fraction"]["ipsweep"] == 1.0



