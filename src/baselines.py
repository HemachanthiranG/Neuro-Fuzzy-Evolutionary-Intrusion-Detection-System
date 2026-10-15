"""Baseline feature selection methods: Mutual Information and Random Forest."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from src.preprocessing import NSLKDDPreprocessor
from src.utils import setup_logger

logger = setup_logger("baselines")


def aggregate_encoded_scores_to_original(
    encoded_scores: np.ndarray,
    preprocessor: NSLKDDPreprocessor,
) -> np.ndarray:
    """Aggregate encoded column importance scores to the 41 original features.

    For numeric features, takes the single column score.
    For one-hot encoded categorical features, takes the mean score across the category columns.
    """
    orig_scores = np.zeros(len(preprocessor.original_feature_names), dtype=np.float64)
    for orig_idx, col_indices in preprocessor.feature_to_encoded_indices.items():
        orig_scores[orig_idx] = float(np.mean(encoded_scores[col_indices]))
    return orig_scores


def select_top_k_mutual_information(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor: NSLKDDPreprocessor,
    k: int,
    seed: int = 42,
    subsample_size: int = 20000,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select top-k original features using Mutual Information ranking.

    Returns:
        Tuple[binary_chromosome of length 41, list of selected feature names, original feature scores]
    """
    logger.info(f"Computing Mutual Information feature importance for top {k} features...")
    # Subsample training data if large for computational efficiency on CPU
    if len(X_train) > subsample_size:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=seed
        )
    else:
        X_sub, y_sub = X_train, y_train

    # Compute MI on all encoded columns
    mi_encoded = mutual_info_classif(X_sub, y_sub, random_state=seed, discrete_features=False)
    orig_scores = aggregate_encoded_scores_to_original(mi_encoded, preprocessor)

    # Pick top k indices
    top_k_orig_indices = np.argsort(orig_scores)[::-1][:k]

    chromosome = np.zeros(len(preprocessor.original_feature_names), dtype=np.int32)
    chromosome[top_k_orig_indices] = 1

    selected_names = [preprocessor.original_feature_names[i] for i in sorted(top_k_orig_indices)]
    logger.info(f"Mutual Information selected {k} features: {selected_names}")
    return chromosome, selected_names, orig_scores


def select_top_k_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor: NSLKDDPreprocessor,
    k: int,
    seed: int = 42,
    subsample_size: int = 25000,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select top-k original features using Random Forest Gini importance ranking.

    Returns:
        Tuple[binary_chromosome of length 41, list of selected feature names, original feature scores]
    """
    logger.info(f"Computing Random Forest feature importance for top {k} features...")
    if len(X_train) > subsample_size:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=seed
        )
    else:
        X_sub, y_sub = X_train, y_train

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_sub, y_sub)

    rf_encoded_importances = rf.feature_importances_
    orig_scores = aggregate_encoded_scores_to_original(rf_encoded_importances, preprocessor)

    top_k_orig_indices = np.argsort(orig_scores)[::-1][:k]

    chromosome = np.zeros(len(preprocessor.original_feature_names), dtype=np.int32)
    chromosome[top_k_orig_indices] = 1

    selected_names = [preprocessor.original_feature_names[i] for i in sorted(top_k_orig_indices)]
    logger.info(f"Random Forest selected {k} features: {selected_names}")
    return chromosome, selected_names, orig_scores
