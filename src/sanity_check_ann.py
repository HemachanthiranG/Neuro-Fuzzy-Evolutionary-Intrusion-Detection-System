"""Sanity check script for Phase 2: ANN module & Baseline ANN."""

import sys
from pathlib import Path
import numpy as np

from src.ann import (
    build_ann_model,
    load_ann_model,
    measure_inference_latency,
    predict_ann,
    save_ann_model,
    train_ann_model,
)
from src.config import load_config
from src.preprocessing import process_dataset_and_generate_eda
from src.utils import set_seeds, setup_logger

logger = setup_logger("ann_sanity_check")


def run_ann_sanity_check() -> bool:
    """Run verification of the ANN module on all 122 features."""
    set_seeds(42)
    cfg = load_config(mode="smoke")

    logger.info("Loading preprocessed dataset...")
    datasets, preprocessor = process_dataset_and_generate_eda(config=cfg, seed=42)

    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    class_weights = datasets["class_weights"]

    input_dim = X_train.shape[1]
    num_classes = len(cfg.data.classes)

    logger.info(f"Building Baseline ANN with input_dim={input_dim}, num_classes={num_classes}...")
    model = build_ann_model(
        input_dim=input_dim,
        config=cfg.ann,
        num_classes=num_classes,
        name="baseline_ann_sanity",
    )

    logger.info(f"Training Baseline ANN for {cfg.ann.epochs} sanity epochs...")
    model, history, train_time = train_ann_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=cfg.ann,
        class_weights=class_weights,
        epochs=3,
        batch_size=256,
        verbose=1,
    )

    # Validate training metrics
    loss_history = history.history["loss"]
    val_loss_history = history.history["val_loss"]
    val_acc_history = history.history["val_accuracy"]

    logger.info(f"Initial loss: {loss_history[0]:.4f} -> Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Validation loss: {val_loss_history[-1]:.4f}, Validation accuracy: {val_acc_history[-1]:.4f}")

    if loss_history[-1] > loss_history[0] * 1.5:
        logger.error("Loss diverged during sanity check.")
        return False

    # Predict
    logger.info("Testing prediction pipeline...")
    pred_val, prob_val = predict_ann(model, X_val[:100])
    assert pred_val.shape == (100,), f"Unexpected pred shape {pred_val.shape}"
    assert prob_val.shape == (100, num_classes), f"Unexpected prob shape {prob_val.shape}"
    assert np.allclose(prob_val.sum(axis=1), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"
    logger.info(f"Sample predictions (first 5): {pred_val[:5]}, True labels: {y_val[:5]}")

    # Latency check
    latency_1k = measure_inference_latency(model, X_val, target_count=1000)
    logger.info(f"Measured inference latency: {latency_1k * 1000:.2f} ms per 1,000 records.")

    # Save & reload verification
    sanity_model_path = cfg.paths.models_dir / "baseline_ann_sanity.keras"
    save_ann_model(model, sanity_model_path)
    assert sanity_model_path.exists(), f"Model file not created at {sanity_model_path}"

    reloaded_model = load_ann_model(sanity_model_path)
    pred_reloaded, prob_reloaded = predict_ann(reloaded_model, X_val[:100])
    assert np.array_equal(pred_val, pred_reloaded), "Reloaded model produced different predictions!"
    assert np.allclose(prob_val, prob_reloaded, atol=1e-5), "Reloaded model probabilities differ!"

    logger.info("ANN module and baseline sanity check PASSED successfully.")
    return True


if __name__ == "__main__":
    success = run_ann_sanity_check()
    if not success:
        sys.exit(1)
