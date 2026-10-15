"""Inference engine supporting test set sample extraction, manual input, and batch CSV predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from src.ann import load_ann_model, predict_ann
from src.config import load_config
from src.fuzzy import FuzzyDecisionLayer
from src.preprocessing import NSLKDDPreprocessor
from src.utils import load_artifact, load_json, setup_logger

logger = setup_logger("app_inference")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RAW_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "KDDTest+.txt"


def _to_path(p: Any, default: Path) -> Path:
    if p is None:
        return default
    path_obj = Path(p)
    if not path_obj.is_absolute():
        candidate = PROJECT_ROOT / path_obj
        if candidate.exists():
            return candidate
    return path_obj


@st.cache_resource
def get_cached_preprocessor(models_dir: Optional[str] = None) -> NSLKDDPreprocessor:
    """Load and cache the preprocessor artifact."""
    m_dir = _to_path(models_dir, DEFAULT_MODELS_DIR)
    p = m_dir / "preprocessor.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {p}. Please run Phase 1 first.")
    return load_artifact(p)


def get_available_seeds(models_dir: Optional[str] = None, results_dir: Optional[str] = None) -> List[int]:
    """Dynamically discover available random seeds from results and saved model artifacts."""
    m_dir = _to_path(models_dir, DEFAULT_MODELS_DIR)
    r_dir = _to_path(results_dir, DEFAULT_RESULTS_DIR)
    seeds = set()
    metrics_path = r_dir / "metrics_all_runs.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            for s in df["seed"].unique():
                seeds.add(int(s))
        except Exception:
            pass

    for p in m_dir.glob("model_*_seed*.keras"):
        try:
            part = p.stem.split("_seed")[-1]
            if part.isdigit():
                seeds.add(int(part))
        except Exception:
            pass

    if not seeds:
        cfg = load_config()
        return sorted(list(cfg.seeds))
    return sorted(list(seeds))
@st.cache_resource
def get_cached_model(model_name: str, seed: Optional[int] = None, models_dir: Optional[str] = None) -> keras.Model:
    """Load and cache a trained Keras model."""
    m_dir = _to_path(models_dir, DEFAULT_MODELS_DIR)
    if seed is None:
        avail = get_available_seeds(models_dir=str(m_dir))
        seed = avail[0] if avail else load_config().seeds[0]

    name_low = model_name.lower()
    if ("fuzzy-input" in name_low and "fuzzy-decision" in name_low) or "proposed" in name_low or "full" in name_low:
        tag = "ga_adaptive_fuzzy_input_ann_fuzzy_decision"
    elif "baseline" in name_low and "fuzzy" in name_low:
        tag = "baseline_ann_fuzzy_decision"
    elif "adaptive" in name_low and "fuzzy-input" in name_low:
        tag = "ga_adaptive_fuzzy_input_ann"
    elif "adaptive" in name_low:
        tag = "ga_adaptive_ann"
    elif "fuzzy-input" in name_low:
        tag = "ga_fuzzy_input_ann"
    elif "ga" in name_low:
        tag = "ga_ann"
    elif "mi" in name_low:
        tag = "mi_ann"
    elif "rf" in name_low:
        tag = "rf_ann"
    else:
        tag = "baseline_ann"

    model_path = m_dir / f"model_{tag}_seed{seed}.keras"
    if not model_path.exists():
        # Fallback to sanity check model if present
        fallback = m_dir / "baseline_ann_sanity.keras"
        if fallback.exists():
            return load_ann_model(fallback)
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please check that models exist in {m_dir}.")
    return load_ann_model(model_path)


@st.cache_resource
def get_cached_decision_layer(
    model_name: str, seed: Optional[int] = None, results_dir: Optional[str] = None
) -> Optional[FuzzyDecisionLayer]:
    """Load and cache the tuned FuzzyDecisionLayer for fuzzy-decision models."""
    name_low = model_name.lower()
    if "fuzzy-decision" not in name_low and "proposed" not in name_low and "full" not in name_low:
        return None

    r_dir = _to_path(results_dir, DEFAULT_RESULTS_DIR)
    cfg = load_config()
    # Base configuration fallback from config.yaml (not hardcoded)
    susp_band = list(getattr(cfg.fuzzy.decision_layer, "suspicious_band", [0.35, 0.65]))

    # Load tuned band from results/fuzzy_decision_stats.json if available
    stats_path = r_dir / "fuzzy_decision_stats.json"
    if stats_path.exists():
        try:
            stats = load_json(stats_path)
            seed_key = str(seed) if seed is not None else next(iter(stats.keys()), "")
            seed_dict = stats.get(seed_key, {})
            for key, val in seed_dict.items():
                if "fuzzy-decision" in key.lower() or "proposed" in key.lower():
                    if "chosen_suspicious_band" in val:
                        susp_band = val["chosen_suspicious_band"]
                        break
        except Exception:
            pass

    return FuzzyDecisionLayer(suspicious_band=list(susp_band))


@st.cache_data
def get_cached_feature_mask(model_name: str, seed: Optional[int] = None, models_dir: Optional[str] = None) -> Optional[List[int]]:
    """Load the feature mask index list for the specified model."""
    name_low = model_name.lower()
    if "baseline" in name_low and "fuzzy" not in name_low:
        return None  # Baseline uses all features

    m_dir = _to_path(models_dir, DEFAULT_MODELS_DIR)
    if seed is None:
        avail = get_available_seeds(models_dir=str(m_dir))
        seed = avail[0] if avail else load_config().seeds[0]

    candidates = []
    if "adaptive" in name_low or "proposed" in name_low or "full" in name_low:
        candidates.extend([
            f"selected_mask_adaptive_ga_seed{seed}.json",
            f"selected_mask_adaptive_seed{seed}.json",
        ])
    elif "mi" in name_low:
        candidates.extend([f"selected_mask_mi_seed{seed}.json"])
    elif "rf" in name_low:
        candidates.extend([f"selected_mask_rf_seed{seed}.json"])
    else:
        candidates.extend([
            f"selected_mask_standard_ga_seed{seed}.json",
            f"selected_mask_standard_seed{seed}.json",
            f"selected_mask_ga_seed{seed}.json",
            f"selected_mask_seed{seed}.json",
        ])

    for c in candidates:
        mask_path = m_dir / c
        if mask_path.exists():
            data = load_json(mask_path)
            return data.get("selected_original_feature_indices", data.get("selected_feature_indices", []))
    return None


@st.cache_data
def load_test_samples_cache(raw_test_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw test set for sample picking in Tab 2."""
    t_path = _to_path(raw_test_path, DEFAULT_RAW_TEST_PATH)
    cfg = load_config()
    df = pd.read_csv(t_path, header=None, names=cfg.data.column_names)
    from src.data_loader import get_attack_to_category_mapping
    attack_map = get_attack_to_category_mapping(cfg.data.attack_mapping)
    df["raw_label"] = df[cfg.data.target_column].astype(str).str.strip().str.lower()
    df["attack_category"] = df["raw_label"].map(attack_map)
    return df


def predict_single_record(
    input_features: Dict[str, Any],
    model_name: str = "GA+ANN",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run inference on a single record dictionary of features."""
    if seed is None:
        avail = get_available_seeds()
        seed = avail[0] if avail else load_config().seeds[0]

    preprocessor = get_cached_preprocessor()
    model = get_cached_model(model_name=model_name, seed=seed)
    ga_mask_indices = get_cached_feature_mask(model_name=model_name, seed=seed)
    decision_layer = get_cached_decision_layer(model_name=model_name, seed=seed)

    name_low = model_name.lower()
    is_fuzzy_input = ("fuzzy-input" in name_low or "proposed" in name_low or "full" in name_low)

    # Ensure all 41 features exist in input_features; fill missing with preprocessor defaults
    complete_row = {}
    for feat in preprocessor.original_feature_names:
        if feat in input_features:
            complete_row[feat] = input_features[feat]
        else:
            complete_row[feat] = preprocessor.feature_defaults.get(feat, 0.0)

    df_single = pd.DataFrame([complete_row])
    X_encoded = preprocessor.transform_features(df_single)

    chrom = np.zeros(len(preprocessor.original_feature_names), dtype=int)
    if ga_mask_indices is not None:
        chrom[ga_mask_indices] = 1
    else:
        chrom[:] = 1

    if is_fuzzy_input:
        X_input = preprocessor.transform_neuro_fuzzy(X_encoded, chrom, input_mode="concat")
    elif ga_mask_indices is not None:
        cols = preprocessor.get_column_indices_for_chromosome(chrom)
        X_input = X_encoded[:, cols]
    else:
        X_input = X_encoded

    pred_idx, probs = predict_ann(model, X_input)
    hard_class = preprocessor.idx_to_class[int(pred_idx[0])]
    confidence = float(probs[0][pred_idx[0]])

    prob_dict = {
        cls_name: float(probs[0][idx])
        for idx, cls_name in preprocessor.idx_to_class.items()
    }

    if decision_layer is not None:
        dec_out = decision_layer.predict(probs)
        fuzzy_class = preprocessor.idx_to_class[int(dec_out["fuzzy_preds"][0])]
        alert_level = dec_out["alert_names"][0]
        severity_score = float(dec_out["severity"][0])
    else:
        fuzzy_class = hard_class
        alert_level = "Attack" if hard_class != "Normal" else "Normal"
        severity_score = float(1.0 - prob_dict.get("Normal", 0.0))

    is_attack = (fuzzy_class != "Normal" or alert_level in ("Attack", "Suspicious"))

    # Compute fuzzy membership degrees for key numeric features
    fuzzy_memberships = {}
    if preprocessor.fuzzy_input_layer is not None and preprocessor.fuzzy_input_layer.is_fitted:
        key_candidates = [
            "count",
            "same_srv_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_src_port_rate",
            "srv_rerror_rate",
            "src_bytes",
            "duration",
        ]
        for feat in key_candidates:
            if feat in preprocessor.fuzzy_input_layer.models_:
                val = float(complete_row.get(feat, 0.0))
                m = preprocessor.fuzzy_input_layer.models_[feat].compute_memberships(np.array([val]))[0]
                fuzzy_memberships[feat] = {
                    "Low": float(m[0]),
                    "Medium": float(m[1]),
                    "High": float(m[2]),
                    "raw_value": val,
                }

    return {
        "predicted_class": hard_class,
        "fuzzy_adjusted_class": fuzzy_class,
        "confidence": confidence,
        "is_attack": is_attack,
        "alert_level": alert_level,
        "severity_score": severity_score,
        "probabilities": prob_dict,
        "fuzzy_memberships": fuzzy_memberships,
        "features_used_count": len(ga_mask_indices) if ga_mask_indices else 41,
        "decision_band": list(decision_layer.suspicious_band) if decision_layer is not None else [0.35, 0.65],
        "is_fuzzy_model": (decision_layer is not None or is_fuzzy_input),
    }


def predict_batch_records(
    df_batch: pd.DataFrame,
    model_name: str = "GA+ANN",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run batch prediction on an uploaded DataFrame matching the 41-feature schema."""
    preprocessor = get_cached_preprocessor()
    model = get_cached_model(model_name=model_name, seed=seed)
    ga_mask_indices = get_cached_feature_mask(model_name=model_name, seed=seed)
    decision_layer = get_cached_decision_layer(model_name=model_name, seed=seed)

    name_low = model_name.lower()
    is_fuzzy_input = ("fuzzy-input" in name_low or "proposed" in name_low or "full" in name_low)

    # Validate required features
    missing_features = [f for f in preprocessor.original_feature_names if f not in df_batch.columns]
    if missing_features:
        raise ValueError(
            f"Uploaded CSV is missing {len(missing_features)} required features: {missing_features[:5]}..."
        )

    X_encoded = preprocessor.transform_features(df_batch)

    chrom = np.zeros(len(preprocessor.original_feature_names), dtype=int)
    if ga_mask_indices is not None:
        chrom[ga_mask_indices] = 1
    else:
        chrom[:] = 1

    if is_fuzzy_input:
        X_input = preprocessor.transform_neuro_fuzzy(X_encoded, chrom, input_mode="concat")
    elif ga_mask_indices is not None:
        cols = preprocessor.get_column_indices_for_chromosome(chrom)
        X_input = X_encoded[:, cols]
    else:
        X_input = X_encoded

    pred_indices, probs = predict_ann(model, X_input)
    hard_classes = [preprocessor.idx_to_class[int(idx)] for idx in pred_indices]
    confidences = [float(probs[i][pred_indices[i]]) for i in range(len(pred_indices))]

    res_df = df_batch.copy()
    res_df["Predicted_Class"] = hard_classes
    res_df["Confidence"] = [round(c, 4) for c in confidences]

    if decision_layer is not None:
        dec_out = decision_layer.predict(probs)
        fuzzy_classes = [preprocessor.idx_to_class[int(idx)] for idx in dec_out["fuzzy_preds"]]
        res_df["Fuzzy_Adjusted_Class"] = fuzzy_classes
        res_df["Alert_Level"] = dec_out["alert_names"]
        res_df["Severity_Score"] = [round(float(s), 4) for s in dec_out["severity"]]
        res_df["Threat_Level"] = [
            "Attack" if a in ("Attack", "Suspicious") or c != "Normal" else "Normal"
            for a, c in zip(dec_out["alert_names"], fuzzy_classes)
        ]
    else:
        res_df["Threat_Level"] = ["Attack" if c != "Normal" else "Normal" for c in hard_classes]

    # Include probabilities for transparency
    for idx, c in preprocessor.idx_to_class.items():
        res_df[f"Prob_{c}"] = [round(float(p[idx]), 4) for p in probs]

    return res_df
