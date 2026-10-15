"""Comprehensive evaluation and metric computation module for NSL-KDD."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from src.ann import measure_inference_latency, predict_ann
from src.utils import save_json, setup_logger

logger = setup_logger("evaluate")


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    split_name: str = "test",
) -> Dict[str, Any]:
    """Compute all required metrics for intrusion detection models."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # One-vs-Rest Macro ROC-AUC if feasible
    roc_auc = None
    try:
        # Check if all classes are present in y_true
        unique_true = np.unique(y_true)
        if len(unique_true) == len(classes):
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")

    # Normal (class 0) vs Any Attack (classes 1..4) Binary Metrics
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)

    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    overall_attack_dr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Per-class metrics and detection rate
    per_class = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    
    for idx, cls_name in enumerate(classes):
        cls_mask_true = (y_true == idx)
        cls_mask_pred = (y_pred == idx)
        support = int(np.sum(cls_mask_true))
        
        p = float(precision_score(cls_mask_true, cls_mask_pred, zero_division=0))
        r = float(recall_score(cls_mask_true, cls_mask_pred, zero_division=0))
        f1 = float(f1_score(cls_mask_true, cls_mask_pred, zero_division=0))

        per_class[cls_name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "detection_rate": round(r, 4),
            "support": support,
        }

    return {
        "split": split_name,
        "accuracy": round(acc, 5),
        "macro_precision": round(macro_p, 5),
        "macro_recall": round(macro_r, 5),
        "macro_f1": round(macro_f1, 5),
        "weighted_precision": round(weighted_p, 5),
        "weighted_recall": round(weighted_r, 5),
        "weighted_f1": round(weighted_f1, 5),
        "roc_auc_ovr": round(roc_auc, 5) if roc_auc is not None else None,
        "false_positive_rate": round(fpr, 5),
        "false_negative_rate": round(fnr, 5),
        "overall_attack_detection_rate": round(overall_attack_dr, 5),
        "binary_confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "per_class": per_class,
    }


def save_and_plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_prefix: Path,
    model_name: str = "Model",
) -> Tuple[Path, Path, Path]:
    """Save raw CM, normalized CM, and a high-resolution heatmap plot."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_prefix.parent / f"{output_prefix.name}.npy"
    norm_path = output_prefix.parent / f"{output_prefix.name}_norm.npy"
    png_path = output_prefix.parent / f"{output_prefix.name}.png"

    np.save(raw_path, cm)
    np.save(norm_path, cm_norm)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=10)
    axes[0].set_ylabel("True Class", fontsize=10)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title(f"{model_name} - Normalized (Recall)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=10)
    axes[1].set_ylabel("True Class", fontsize=10)

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    return raw_path, norm_path, png_path


def evaluate_model_pipeline(
    model: keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    model_name: str,
    seed: int,
    cm_dir: Path,
    training_time: float,
    num_orig_features: int,
    num_encoded_features: int,
    model_path: Optional[Path] = None,
    fuzzy_decision_layer: Optional[Any] = None,
    beta_fpr_penalty: float = 0.30,
    workload_penalty: float = 0.10,
    false_alarm_penalty: float = 0.30,
    raw_attacks_val: Optional[np.ndarray] = None,
    raw_attacks_test: Optional[np.ndarray] = None,
    reused_val_prob: Optional[np.ndarray] = None,
    reused_test_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute evaluation on both Validation and Test sets and record all metadata."""
    # Raw ANN probabilistic predictions (reused if provided for strict decision layer isolation)
    if reused_val_prob is not None:
        val_prob = reused_val_prob
        val_pred = np.argmax(val_prob, axis=1).astype(np.int32)
    else:
        val_pred, val_prob = predict_ann(model, X_val)

    if reused_test_prob is not None:
        test_prob = reused_test_prob
        test_pred = np.argmax(test_prob, axis=1).astype(np.int32)
    else:
        test_pred, test_prob = predict_ann(model, X_test)

    chosen_band = None
    decision_stats = None
    suspicious_stats_val = None
    suspicious_stats_test = None
    alert_metrics_test = None
    alert_metrics_val = None

    # If Fuzzy Decision Layer is provided, tune strictly on validation and apply
    if fuzzy_decision_layer is not None:
        tune_stats = fuzzy_decision_layer.tune_on_validation(
            val_probs=val_prob,
            y_val=y_val,
            false_alarm_penalty=false_alarm_penalty,
            workload_penalty=workload_penalty,
            beta_fpr_penalty=beta_fpr_penalty,
            classes=classes,
            raw_attacks_val=raw_attacks_val,
        )
        val_decision = fuzzy_decision_layer.predict(val_prob, classes=classes)
        test_decision = fuzzy_decision_layer.predict(test_prob, classes=classes)

        val_pred = np.argmax(val_prob, axis=1).astype(np.int32)
        test_pred = np.argmax(test_prob, axis=1).astype(np.int32)

        chosen_band = fuzzy_decision_layer.suspicious_band
        decision_stats = tune_stats
        suspicious_stats_val = tune_stats.get("suspicious_band_validation_stats")
        suspicious_stats_test = fuzzy_decision_layer.compute_suspicious_statistics(
            test_decision["alert_indices"], y_test, classes=classes
        )
        alert_metrics_val = fuzzy_decision_layer.compute_alert_metrics(
            val_decision["alert_indices"],
            y_val,
            raw_attacks=raw_attacks_val,
        )
        alert_metrics_test = fuzzy_decision_layer.compute_alert_metrics(
            test_decision["alert_indices"],
            y_test,
            raw_attacks=raw_attacks_test,
        )
        decision_stats["test_alert_metrics"] = alert_metrics_test
        decision_stats["val_alert_metrics"] = alert_metrics_val
        decision_stats["fuzzy_adjusted_differ_count_val"] = int(np.sum(val_decision["fuzzy_preds"] != val_pred))
        decision_stats["fuzzy_adjusted_differ_count_test"] = int(np.sum(test_decision["fuzzy_preds"] != test_pred))
    else:
        # For non-fuzzy models, alert metrics correspond to standard binary threshold (alert 2 = attack, alert 0 = normal)
        val_alert_indices = np.where(val_pred != 0, 2, 0).astype(np.int32)
        test_alert_indices = np.where(test_pred != 0, 2, 0).astype(np.int32)
        from src.fuzzy import FuzzyDecisionLayer
        dummy_layer = FuzzyDecisionLayer()
        alert_metrics_val = dummy_layer.compute_alert_metrics(val_alert_indices, y_val, raw_attacks=raw_attacks_val)
        alert_metrics_test = dummy_layer.compute_alert_metrics(test_alert_indices, y_test, raw_attacks=raw_attacks_test)

    # Metrics
    val_metrics = compute_comprehensive_metrics(y_val, val_pred, val_prob, classes, split_name="validation")
    test_metrics = compute_comprehensive_metrics(y_test, test_pred, test_prob, classes, split_name="test")

    # Latency
    latency_sec = measure_inference_latency(model, X_test, target_count=1000)

    # Model size
    model_size_kb = 0.0
    if model_path and model_path.exists():
        model_size_kb = round(model_path.stat().st_size / 1024.0, 2)

    # Save test confusion matrix
    cm_prefix = cm_dir / f"cm_{model_name.lower().replace('+', '_').replace(' ', '_').replace('(', '_').replace(')', '_')}_seed{seed}"
    save_and_plot_confusion_matrix(y_test, test_pred, classes, cm_prefix, model_name=f"{model_name} (Seed {seed})")

    result = {
        "model_name": model_name,
        "seed": seed,
        "num_features_original": num_orig_features,
        "num_features_encoded": num_encoded_features,
        "training_time_sec": round(training_time, 2),
        "inference_time_per_1000_ms": round(latency_sec * 1000.0, 2),
        "model_size_kb": model_size_kb,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_false_positive_rate": val_metrics["false_positive_rate"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_precision": test_metrics["weighted_precision"],
        "test_weighted_recall": test_metrics["weighted_recall"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_roc_auc_ovr": test_metrics["roc_auc_ovr"],
        "test_false_positive_rate": test_metrics["false_positive_rate"],
        "test_false_negative_rate": test_metrics["false_negative_rate"],
        "test_attack_detection_rate": test_metrics["overall_attack_detection_rate"],
        "chosen_suspicious_band": chosen_band,
        "per_class_test": test_metrics["per_class"],
        "decision_stats": decision_stats,
        "suspicious_stats_val": suspicious_stats_val,
        "suspicious_stats_test": suspicious_stats_test,
        "alert_metrics_test": alert_metrics_test,
        "alert_metrics_val": alert_metrics_val,
        "val_prob": val_prob,
        "test_prob": test_prob,
    }
    if alert_metrics_test:
        result["test_attack_recall_flagged"] = alert_metrics_test["attack_recall_flagged"]
        result["test_attack_alert_precision"] = alert_metrics_test["attack_alert_precision"]
        result["test_normal_flagged_rate"] = alert_metrics_test["normal_flagged_rate"]
        result["test_analyst_workload_rate"] = alert_metrics_test["analyst_workload_rate"]

    return result
