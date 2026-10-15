"""Unit and integration tests for NSL-KDD Feature Selection Pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.components.inference import predict_batch_records, predict_single_record
from src.config import load_config
from src.data_loader import get_attack_to_category_mapping, load_raw_nsl_kdd
from src.ga import GeneticFeatureSelector
from src.preprocessing import NSLKDDPreprocessor
from src.utils import load_artifact


@pytest.fixture(scope="session")
def app_config():
    return load_config(mode="smoke")


@pytest.fixture(scope="session")
def preprocessor():
    p = Path("models/preprocessor.joblib")
    assert p.exists(), "Fitted preprocessor model does not exist. Run Phase 1 first."
    return load_artifact(p)


def test_label_mapping_completeness(app_config):
    """Verify that all attack names in KDDTrain+ and KDDTest+ are mapped to 5 classes."""
    attack_map = get_attack_to_category_mapping(app_config.data.attack_mapping)

    df_train = pd.read_csv("data/raw/KDDTrain+.txt", header=None)
    df_test = pd.read_csv("data/raw/KDDTest+.txt", header=None)

    train_labels = set(df_train.iloc[:, 41].astype(str).str.strip().str.lower().unique())
    test_labels = set(df_test.iloc[:, 41].astype(str).str.strip().str.lower().unique())
    all_raw_labels = train_labels.union(test_labels)

    unmapped = all_raw_labels - set(attack_map.keys())
    assert len(unmapped) == 0, f"Found unmapped labels in NSL-KDD: {unmapped}"

    # Verify that all 5 target classes exist in mapping outputs
    mapped_categories = set(attack_map.values())
    expected_categories = {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert mapped_categories == expected_categories


def test_no_label_in_features(preprocessor):
    """Ensure neither 'label' nor 'difficulty' nor target columns leak into feature matrix."""
    forbidden = {"label", "difficulty", "attack_category", "raw_label"}

    for name in preprocessor.original_feature_names:
        assert name.lower() not in forbidden, f"Forbidden target/difficulty column found in features: {name}"

    for enc_name in preprocessor.encoded_feature_names:
        assert not any(enc_name.startswith(f) for f in forbidden), f"Forbidden target in encoded features: {enc_name}"

    assert len(preprocessor.original_feature_names) == 41, f"Expected 41 features, found {len(preprocessor.original_feature_names)}"


def test_scaler_fit_on_train_only(preprocessor):
    """Verify that scaler parameters match training data dimensions and are properly fitted."""
    assert preprocessor.is_fitted, "Preprocessor should be marked as fitted."
    assert hasattr(preprocessor.scaler, "mean_"), "StandardScaler should have fitted mean_."
    assert len(preprocessor.scaler.mean_) == len(preprocessor.numeric_features)
    assert len(preprocessor.numeric_features) == 38, f"Expected 38 numeric features, got {len(preprocessor.numeric_features)}"


def test_ga_repair_keeps_at_least_one_feature(app_config, preprocessor):
    """Verify that GA chromosome repair enforces >= 1 selected feature."""
    ga = GeneticFeatureSelector(
        ga_config=app_config.ga,
        ann_config=app_config.ann,
        preprocessor=preprocessor,
        seed=42,
    )

    all_zeros = np.zeros(41, dtype=int)
    repaired = ga.repair_chromosome(all_zeros)

    assert np.sum(repaired) >= 1, "Repaired chromosome must have at least one active feature."
    assert len(repaired) == 41, "Repaired chromosome length must remain 41."

    # Non-zero chromosome should be unchanged
    non_zero = np.zeros(41, dtype=int)
    non_zero[5] = 1
    assert np.array_equal(ga.repair_chromosome(non_zero), non_zero)


def test_feature_mask_to_columns_mapping_correctness(preprocessor):
    """Verify that 41 original features partition all encoded columns with no overlap or omissions."""
    all_mapped_cols = []
    for orig_idx in range(len(preprocessor.original_feature_names)):
        cols = preprocessor.feature_to_encoded_indices[orig_idx]
        assert len(cols) >= 1, f"Original feature {orig_idx} must map to at least 1 encoded column."
        all_mapped_cols.extend(cols)

    # All encoded columns must be present exactly once
    sorted_mapped = sorted(all_mapped_cols)
    expected_cols = list(range(len(preprocessor.encoded_feature_names)))
    assert sorted_mapped == expected_cols, "Feature-to-encoded column mapping is incomplete or contains overlaps."

    # Verify categorical features map to multiple columns
    for cat_feat in preprocessor.categorical_features:
        cols = preprocessor.feature_name_to_encoded_indices[cat_feat]
        assert len(cols) > 1, f"Categorical feature {cat_feat} should expand to multiple one-hot columns."


def test_prediction_pipeline_shape_check():
    """Verify single and batch prediction pipelines return valid shapes, confidences, and labels."""
    cfg = load_config()
    df_test = pd.read_csv("data/raw/KDDTest+.txt", header=None, names=cfg.data.column_names, nrows=10)

    # 1. Single sample prediction
    sample_dict = df_test.iloc[0].to_dict()
    res = predict_single_record(sample_dict, model_name="GA+ANN", seed=42)

    assert "predicted_class" in res
    assert res["predicted_class"] in cfg.data.classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == len(cfg.data.classes)
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-4)

    # 2. Batch prediction
    batch_df = df_test.drop(columns=["label", "difficulty"], errors="ignore")
    batch_res = predict_batch_records(batch_df, model_name="GA+ANN", seed=42)

    assert len(batch_res) == 10
    assert "Predicted_Class" in batch_res.columns
    assert "Confidence" in batch_res.columns
    assert "Threat_Level" in batch_res.columns
    assert all(c in cfg.data.classes for c in batch_res["Predicted_Class"])
    assert all(t in ["Attack", "Normal"] for t in batch_res["Threat_Level"])
