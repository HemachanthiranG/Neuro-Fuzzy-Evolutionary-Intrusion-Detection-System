"""Artificial Neural Network (ANN) architecture and training routines for NSL-KDD."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

from src.config import ANNConfig
from src.utils import setup_logger

logger = setup_logger("ann")


def build_ann_model(
    input_dim: int,
    config: ANNConfig,
    num_classes: int = 5,
    name: str = "ids_ann",
) -> keras.Model:
    """Build a configurable multi-class Keras MLP model.

    Architecture:
    Input(input_dim) -> [Dense(units) -> BatchNorm? -> Dropout(rate)] * N -> Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = inputs

    for idx, (units, dropout, act) in enumerate(
        zip(config.hidden_layers, config.dropouts, config.activations)
    ):
        x = layers.Dense(units, activation=act, name=f"{name}_dense_{idx + 1}")(x)
        if config.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn_{idx + 1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"{name}_dropout_{idx + 1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name=f"{name}_output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ANNConfig,
    class_weights: Optional[Dict[int, float]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[keras.Model, keras.callbacks.History, float]:
    """Train the ANN model with early stopping, learning rate reduction, and optional class weights."""
    n_epochs = epochs if epochs is not None else config.epochs
    bs = batch_size if batch_size is not None else config.batch_size

    cb_list: List[callbacks.Callback] = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=verbose,
        ),
    ]

    fit_weights = class_weights if config.use_class_weights else None

    start_time = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=n_epochs,
        batch_size=bs,
        callbacks=cb_list,
        class_weight=fit_weights,
        verbose=verbose,
    )
    training_time = time.perf_counter() - start_time
    logger.info(
        f"Model '{model.name}' trained for {len(history.history['loss'])} epochs in {training_time:.2f}s."
    )
    return model, history, training_time


def predict_ann(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate discrete class predictions and probability distributions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (pred_classes of shape (N,), probabilities of shape (N, num_classes))
    """
    probabilities = model.predict(X, batch_size=batch_size, verbose=0)
    pred_classes = np.argmax(probabilities, axis=1).astype(np.int32)
    return pred_classes, probabilities


def measure_inference_latency(
    model: keras.Model,
    X_sample: np.ndarray,
    target_count: int = 1000,
    warmup_runs: int = 3,
    eval_runs: int = 10,
) -> float:
    """Measure inference latency in seconds per target_count (default 1000) records."""
    n_available = len(X_sample)
    if n_available == 0:
        return 0.0

    if n_available >= target_count:
        test_batch = X_sample[:target_count]
    else:
        # Tile rows to reach target_count
        repeat_factor = int(np.ceil(target_count / n_available))
        test_batch = np.tile(X_sample, (repeat_factor, 1))[:target_count]

    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(test_batch, batch_size=target_count, verbose=0)

    # Timed runs
    times = []
    for _ in range(eval_runs):
        t0 = time.perf_counter()
        _ = model.predict(test_batch, batch_size=target_count, verbose=0)
        times.append(time.perf_counter() - t0)

    mean_time = float(np.mean(times))
    return mean_time


def save_ann_model(model: keras.Model, filepath: Path) -> None:
    """Save model to Keras .keras format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    logger.info(f"Model saved to {filepath}")


def load_ann_model(filepath: Path) -> keras.Model:
    """Load model from Keras .keras format."""
    model = keras.models.load_model(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model
