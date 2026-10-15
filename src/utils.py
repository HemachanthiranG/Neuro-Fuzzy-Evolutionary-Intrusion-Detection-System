"""Utility functions for reproducibility, logging, I/O, and timing."""

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union
import joblib
import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Avoid non-deterministic ops if possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def setup_logger(name: str = "ids_experiment", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a unified logger for both console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save serializable data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, file_path: Path) -> None:
    """Save an object using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


def load_artifact(file_path: Path) -> Any:
    """Load an object using joblib."""
    return joblib.load(file_path)


@contextmanager
def timer(description: str = "Task", logger: Optional[logging.Logger] = None) -> Generator[Dict[str, float], None, None]:
    """Measure elapsed execution time of a block of code."""
    res = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield res
    finally:
        elapsed = time.perf_counter() - start
        res["elapsed"] = elapsed
        msg = f"{description} finished in {elapsed:.3f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)
