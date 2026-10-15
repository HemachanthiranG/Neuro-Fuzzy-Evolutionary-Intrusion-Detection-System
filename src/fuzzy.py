"""Custom Fuzzy Logic components for NSL-KDD Feature Selection and Classification.

Implements from scratch in pure NumPy:
1. 1D Fuzzy C-Means (FCM) clustering with robust quantile fallback.
2. Vectorized Gaussian Membership Function input layer.
3. Neuro-fuzzy input transformation for selected original features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import FuzzyInputConfig
from src.utils import save_json, setup_logger

logger = setup_logger("fuzzy")


class FuzzyCMeans1D:
    """1D Fuzzy C-Means (FCM) clustering for numeric feature partitioning into fuzzy sets."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 100,
        error: float = 1e-4,
        min_variance: float = 1e-5,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.min_variance = min_variance
        self.random_state = random_state

        self.centers_: Optional[np.ndarray] = None
        self.sigmas_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, x: np.ndarray) -> "FuzzyCMeans1D":
        """Fit fuzzy clusters on a 1D numeric array."""
        x = np.asarray(x, dtype=np.float64).flatten()
        n_samples = len(x)

        if n_samples == 0:
            raise ValueError("Cannot fit FuzzyCMeans on empty array.")

        std_dev = float(np.std(x))
        unique_vals = np.unique(x)

        # Fallback for zero-variance, binary, or extreme discrete constants
        if std_dev < self.min_variance or len(unique_vals) <= 2:
            self._quantile_fallback(x)
            self.is_fitted = True
            return self

        # Initialize cluster centers at evenly spaced quantiles
        quantiles = np.linspace(10, 90, self.n_clusters)
        centers = np.percentile(x, quantiles)
        centers = np.sort(centers)

        # Ensure centers are distinct
        for i in range(1, len(centers)):
            if centers[i] <= centers[i - 1]:
                centers[i] = centers[i - 1] + 1e-3

        eps = 1e-8
        power = 2.0 / (self.m - 1.0)

        for _ in range(self.max_iter):
            # Compute distance matrix: shape (n_samples, n_clusters)
            dists = np.abs(x[:, np.newaxis] - centers[np.newaxis, :]) + eps

            # Compute memberships
            # u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
            inv_dists = 1.0 / dists
            denom = np.sum(inv_dists ** power, axis=1, keepdims=True)
            memberships = (inv_dists ** power) / denom

            # Update centers
            u_m = memberships ** self.m
            new_centers = np.sum(u_m * x[:, np.newaxis], axis=0) / (np.sum(u_m, axis=0) + eps)
            new_centers = np.sort(new_centers)

            center_shift = np.max(np.abs(new_centers - centers))
            centers = new_centers
            if center_shift < self.error:
                break

        # Compute Gaussian sigmas from fuzzy memberships:
        # sigma_k = sqrt( sum(u_ik^m * (x_i - v_k)^2) / sum(u_ik^m) )
        dists_sq = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
        weighted_sq_diff = np.sum(u_m * dists_sq, axis=0)
        sum_weights = np.sum(u_m, axis=0) + eps
        sigmas = np.sqrt(weighted_sq_diff / sum_weights)

        # Floor sigmas to prevent zero division
        min_sigma = max(std_dev * 0.1, 1e-3)
        sigmas = np.maximum(sigmas, min_sigma)

        self.centers_ = centers
        self.sigmas_ = sigmas
        self.is_fitted = True
        return self

    def _quantile_fallback(self, x: np.ndarray) -> None:
        """Fallback initialization for near-constant or binary features."""
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        if np.isclose(min_val, max_val):
            # Constant feature: spread centers slightly around constant
            center_val = min_val
            spread = 0.5 if abs(center_val) < 1e-4 else abs(center_val) * 0.2
            self.centers_ = np.linspace(center_val - spread, center_val + spread, self.n_clusters)
            self.sigmas_ = np.full(self.n_clusters, max(spread * 0.5, 0.1))
        else:
            # Low-variance or binary: spread across data span
            self.centers_ = np.linspace(min_val, max_val, self.n_clusters)
            span = max_val - min_val
            self.sigmas_ = np.full(self.n_clusters, max(span / (self.n_clusters * 1.5), 0.1))

    def compute_memberships(self, x: np.ndarray) -> np.ndarray:
        """Compute Gaussian membership degrees in [0, 1] for input 1D array."""
        if not self.is_fitted:
            raise RuntimeError("FuzzyCMeans1D must be fitted before computing memberships.")

        x = np.asarray(x, dtype=np.float64).flatten()
        # Shape: (n_samples, n_clusters)
        diff_sq = (x[:, np.newaxis] - self.centers_[np.newaxis, :]) ** 2
        denom = 2.0 * (self.sigmas_[np.newaxis, :] ** 2)
        memberships = np.exp(-diff_sq / denom)
        # Guarantee numerical bounds [0.0, 1.0]
        return np.clip(memberships, 0.0, 1.0).astype(np.float32)


class FuzzyInputLayer:
    """Fuzzy input transformation layer applying 1D Gaussian membership functions across numeric features."""

    def __init__(
        self,
        config: Optional[FuzzyInputConfig] = None,
        term_names: Optional[List[str]] = None,
    ):
        self.config = config or FuzzyInputConfig()
        self.num_terms = self.config.num_terms
        self.input_mode = self.config.input_mode
        self.term_names = term_names or (
            ["Low", "Medium", "High"] if self.num_terms == 3 else ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
        )

        self.models_: Dict[str, FuzzyCMeans1D] = {}
        self.numeric_feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X_train_numeric: np.ndarray, numeric_feature_names: List[str]) -> "FuzzyInputLayer":
        """Fit fuzzy partition models for each numeric feature using training split only."""
        if X_train_numeric.shape[1] != len(numeric_feature_names):
            raise ValueError("Dimension mismatch between numeric data and feature name list.")

        self.numeric_feature_names = list(numeric_feature_names)
        self.models_ = {}

        for col_idx, feat_name in enumerate(self.numeric_feature_names):
            col_data = X_train_numeric[:, col_idx]
            fcm = FuzzyCMeans1D(
                n_clusters=self.num_terms,
                m=self.config.fcm_m,
                max_iter=self.config.fcm_max_iter,
                error=self.config.fcm_error,
                min_variance=self.config.min_variance,
            )
            fcm.fit(col_data)
            self.models_[feat_name] = fcm

        self.is_fitted = True
        logger.info(
            f"Fitted FuzzyInputLayer across {len(self.numeric_feature_names)} numeric features ({self.num_terms} terms each)."
        )
        return self

    def transform(self, X_numeric: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Compute membership degrees for a set of numeric features."""
        if not self.is_fitted:
            raise RuntimeError("FuzzyInputLayer must be fitted before transform.")

        membership_blocks = []
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in self.models_:
                raise ValueError(f"Feature '{feat_name}' not fitted in FuzzyInputLayer.")

            col_data = X_numeric[:, feat_idx]
            m_block = self.models_[feat_name].compute_memberships(col_data)
            membership_blocks.append(m_block)

        if not membership_blocks:
            return np.empty((len(X_numeric), 0), dtype=np.float32)

        return np.hstack(membership_blocks).astype(np.float32)

    def transform_selected_features(
        self,
        X_encoded_full: np.ndarray,
        preprocessor: Any,
        selected_chromosome: np.ndarray,
        input_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Transform encoded feature array into neuro-fuzzy representation based on selected chromosome.

        Supports two modes:
        - 'concat': [Scaled Selected Numeric, One-Hot Categorical, Membership Degrees]
        - 'membership_only': [One-Hot Categorical, Membership Degrees]
        """
        mode = input_mode if input_mode is not None else self.input_mode
        active_indices = [i for i, active in enumerate(selected_chromosome) if active]

        # Partition active original features into numeric and categorical
        active_numeric_names = [
            preprocessor.original_feature_names[i]
            for i in active_indices
            if preprocessor.original_feature_names[i] in preprocessor.numeric_features
        ]

        # Get column indices in X_encoded_full for selected features
        selected_col_indices = preprocessor.get_column_indices_for_chromosome(selected_chromosome)
        X_selected_encoded = X_encoded_full[:, selected_col_indices]

        # Extract numeric columns from full encoded matrix (numeric columns are the first N columns)
        num_len = len(preprocessor.numeric_features)
        # Compute fuzzy memberships for active numeric features
        if active_numeric_names:
            active_num_indices = [
                preprocessor.numeric_features.index(f) for f in active_numeric_names
            ]
            X_active_numeric = X_encoded_full[:, active_num_indices]
            memberships = self.transform(X_active_numeric, active_numeric_names)
        else:
            memberships = np.empty((len(X_encoded_full), 0), dtype=np.float32)

        if mode == "concat":
            # Concatenate selected encoded features + membership degrees
            return np.hstack([X_selected_encoded, memberships]).astype(np.float32)
        elif mode == "membership_only":
            # Categorical columns only + memberships
            cat_col_indices = [
                idx for idx in selected_col_indices if idx >= num_len
            ]
            if cat_col_indices:
                X_cat = X_encoded_full[:, cat_col_indices]
                return np.hstack([X_cat, memberships]).astype(np.float32)
            else:
                return memberships.astype(np.float32)
        else:
            raise ValueError(f"Unknown input_mode '{mode}'. Expected 'concat' or 'membership_only'.")

    def export_params_to_dict(self) -> Dict[str, Any]:
        """Export learned centers and sigmas for visualization and documentation."""
        params = {}
        for feat, model in self.models_.items():
            params[feat] = {
                "terms": self.term_names,
                "centers": [round(float(c), 4) for c in model.centers_],
                "sigmas": [round(float(s), 4) for s in model.sigmas_],
            }
        return params

    def save_params_json(self, output_path: Path) -> None:
        """Save fuzzy parameters to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.export_params_to_dict(), output_path)
        logger.info(f"Saved fuzzy membership parameters to {output_path}")


def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """Evaluate standard triangular membership degree.

    a: left base, b: center peak (membership=1.0), c: right base.
    """
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def left_shoulder_mf(x: float, a: float, b: float) -> float:
    """Evaluate left shoulder membership degree (1.0 for x <= a, declining to 0 at b)."""
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a) if b > a else 1.0


def right_shoulder_mf(x: float, a: float, b: float) -> float:
    """Evaluate right shoulder membership degree (0 for x <= a, rising to 1.0 for x >= b)."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a) if b > a else 1.0


class FuzzyGAController:
    """Mamdani Fuzzy Logic Controller for dynamic GA parameter adaptation.

    Inputs:
    1. Population Diversity: normalized mean pairwise Hamming distance in [0, 1].
    2. Stagnation: normalized consecutive generations without best fitness improvement in [0, 1].

    Outputs:
    1. Adapted Mutation Rate (pm in [min_pm, max_pm])
    2. Adapted Crossover Rate (pc in [min_pc, max_pc])

    Inference:
    - 9 Mamdani rules with triangular/shoulder memberships.
    - Min t-norm for rule antecedent activation.
    - Centroid / Weighted-average defuzzification.
    """

    def __init__(
        self,
        min_mutation_rate: float = 0.005,
        max_mutation_rate: float = 0.10,
        min_crossover_rate: float = 0.60,
        max_crossover_rate: float = 0.95,
        max_stagnation_patience: int = 6,
    ):
        self.min_pm = min_mutation_rate
        self.max_pm = max_mutation_rate
        self.min_pc = min_crossover_rate
        self.max_pc = max_crossover_rate
        self.max_stagnation_patience = max(1, max_stagnation_patience)

        # Output term centers
        self.pm_centers = {
            "Low": self.min_pm + 0.15 * (self.max_pm - self.min_pm),
            "Medium": self.min_pm + 0.45 * (self.max_pm - self.min_pm),
            "High": self.max_pm - 0.10 * (self.max_pm - self.min_pm),
        }
        self.pc_centers = {
            "Low": self.min_pc + 0.15 * (self.max_pc - self.min_pc),
            "Medium": self.min_pc + 0.50 * (self.max_pc - self.min_pc),
            "High": self.max_pc - 0.08 * (self.max_pc - self.min_pc),
        }

        # 9 Mamdani Rules: (div_term, stag_term, pm_term, pc_term, description)
        self.rules = [
            ("Low", "High", "High", "Low", "Low diversity & High stagnation -> High mutation, Low crossover (escape local optimum)"),
            ("Low", "Medium", "High", "Medium", "Low diversity & Medium stagnation -> High mutation, Medium crossover"),
            ("Low", "Low", "Medium", "High", "Low diversity & Low stagnation -> Medium mutation, High crossover (converging well)"),
            ("Medium", "High", "High", "Medium", "Medium diversity & High stagnation -> High mutation, Medium crossover (boost search)"),
            ("Medium", "Medium", "Medium", "High", "Medium diversity & Medium stagnation -> Medium mutation, High crossover (steady state)"),
            ("Medium", "Low", "Low", "High", "Medium diversity & Low stagnation -> Low mutation, High crossover (exploit discovery)"),
            ("High", "High", "Medium", "Medium", "High diversity & High stagnation -> Medium mutation, Medium crossover (guide scattered search)"),
            ("High", "Medium", "Low", "High", "High diversity & Medium stagnation -> Low mutation, High crossover (continue healthy exploration)"),
            ("High", "Low", "Low", "High", "High diversity & Low stagnation -> Low mutation, High crossover (exploit highly diverse progress)"),
        ]

    @staticmethod
    def compute_population_diversity(population: List[np.ndarray]) -> float:
        """Compute normalized mean pairwise Hamming distance across population in [0, 1].

        Uses vectorized O(N*L) gene-count formulation:
        For binary chromosomes, the number of differing pairs at gene k is S_k * (N - S_k),
        where S_k is the sum of 1s at gene k across all individuals.
        """
        n = len(population)
        if n <= 1:
            return 0.0

        pop_matrix = np.array(population, dtype=np.int32)
        n_individuals, n_genes = pop_matrix.shape

        s = np.sum(pop_matrix, axis=0)  # Count of 1s per gene
        total_diffs = np.sum(s * (n_individuals - s))
        num_pairs = (n_individuals * (n_individuals - 1)) / 2.0
        mean_hamming = total_diffs / (num_pairs * n_genes)

        # Scale by 2.0 since max expected Hamming distance for random independent binary genes is 0.5
        norm_diversity = min(1.0, max(0.0, float(mean_hamming * 2.0)))
        return norm_diversity

    def compute_stagnation(self, no_improvement_count: int) -> float:
        """Compute normalized stagnation metric in [0, 1]."""
        return min(1.0, max(0.0, float(no_improvement_count) / float(self.max_stagnation_patience)))

    def fuzzify_inputs(self, diversity: float, stagnation: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute membership degrees for diversity and stagnation."""
        d = min(1.0, max(0.0, diversity))
        s = min(1.0, max(0.0, stagnation))

        # Diversity fuzzy terms
        mu_div = {
            "Low": left_shoulder_mf(d, 0.0, 0.45),
            "Medium": triangular_mf(d, 0.15, 0.50, 0.85),
            "High": right_shoulder_mf(d, 0.55, 1.0),
        }

        # Stagnation fuzzy terms
        mu_stag = {
            "Low": left_shoulder_mf(s, 0.0, 0.40),
            "Medium": triangular_mf(s, 0.15, 0.50, 0.85),
            "High": right_shoulder_mf(s, 0.50, 1.0),
        }

        return mu_div, mu_stag

    def evaluate(
        self,
        diversity: float,
        stagnation: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Evaluate Mamdani controller and return adapted (mutation_rate, crossover_rate, debug_info)."""
        mu_div, mu_stag = self.fuzzify_inputs(diversity, stagnation)

        total_weight = 0.0
        pm_weighted_sum = 0.0
        pc_weighted_sum = 0.0
        firing_strengths = []

        for div_term, stag_term, pm_term, pc_term, desc in self.rules:
            # Min t-norm for antecedent
            weight = min(mu_div[div_term], mu_stag[stag_term])
            firing_strengths.append(weight)

            if weight > 0.0:
                total_weight += weight
                pm_weighted_sum += weight * self.pm_centers[pm_term]
                pc_weighted_sum += weight * self.pc_centers[pc_term]

        if total_weight > 1e-7:
            adapted_pm = pm_weighted_sum / total_weight
            adapted_pc = pc_weighted_sum / total_weight
        else:
            # Fallback to midpoints
            adapted_pm = (self.min_pm + self.max_pm) / 2.0
            adapted_pc = (self.min_pc + self.max_pc) / 2.0

        # Enforce strict bounds
        adapted_pm = float(np.clip(adapted_pm, self.min_pm, self.max_pm))
        adapted_pc = float(np.clip(adapted_pc, self.min_pc, self.max_pc))

        debug_info = {
            "diversity": round(diversity, 4),
            "stagnation": round(stagnation, 4),
            "adapted_pm": round(adapted_pm, 5),
            "adapted_pc": round(adapted_pc, 4),
            "active_rules_count": int(np.sum(np.array(firing_strengths) > 0.0)),
        }

        return adapted_pm, adapted_pc, debug_info

    def export_rules_dict(self) -> List[Dict[str, Any]]:
        """Export readable rule definitions for documentation and UI presentation."""
        rules_list = []
        for i, (d_t, s_t, pm_t, pc_t, desc) in enumerate(self.rules, start=1):
            rules_list.append({
                "rule_id": i,
                "antecedent": f"IF Diversity is {d_t} AND Stagnation is {s_t}",
                "consequent": f"THEN MutationRate is {pm_t} AND CrossoverRate is {pc_t}",
                "description": desc,
            })
        return rules_list


class FuzzyDecisionLayer:
    """Interpretable post-ANN Fuzzy Decision Layer for 3-level alert categorization and false positive mitigation.

    Inputs per record:
    1. Confidence: max class probability in [0.2, 1.0]
    2. Probability of Normal: P(Normal) in [0.0, 1.0]
    3. Prediction Entropy: normalized Shannon entropy in [0.0, 1.0]

    Outputs per record:
    1. Attack Severity: continuous severity score in [0.0, 1.0]
    2. 3-Level Alert: Normal (0), Suspicious (1), Attack (2)
    3. Fuzzy-Adjusted 5-Class Prediction: adjusted label minimizing false positives and rare-class false negatives.

    Validation Guarantee:
    - Suspicious-band cutoffs and thresholds are tuned strictly on validation splits only.
    - Test evaluation evaluates once without parameter feedback.
    """

    def __init__(
        self,
        suspicious_band: Optional[List[float]] = None,
        alert_levels: Optional[List[str]] = None,
    ):
        self.suspicious_band = suspicious_band if suspicious_band is not None else [0.35, 0.65]
        self.alert_levels = alert_levels if alert_levels is not None else ["Normal", "Suspicious", "Attack"]
        self.is_tuned = False
        self.tuning_stats: Dict[str, Any] = {}

        # Severity output term centers: Normal=0.10, Suspicious=0.50, Attack=0.90
        self.severity_centers = {
            "Normal": 0.10,
            "Suspicious": 0.50,
            "Attack": 0.90,
        }

        # Interpretable Mamdani rules: (conf_term, norm_term, ent_term, severity_term, description)
        self.rules = [
            ("High", "High", None, "Normal", "High confidence in Normal -> Normal alert (Low severity)"),
            ("Medium", "High", "Low", "Normal", "Moderate Normal prob with low entropy -> Normal alert"),
            ("High", "Low", None, "Attack", "High confidence attack -> Attack alert (High severity)"),
            ("Medium", "Low", "Low", "Attack", "Moderate attack prob with low entropy -> Attack alert"),
            ("Low", None, "High", "Suspicious", "Low confidence & High entropy -> Suspicious alert"),
            ("Medium", None, "High", "Suspicious", "Medium confidence & High entropy -> Suspicious alert"),
            (None, "Medium", None, "Suspicious", "Ambiguous Normal probability -> Suspicious alert"),
            (None, None, "High", "Suspicious", "High prediction entropy -> Suspicious alert"),
            ("Medium", "Low", "Medium", "Attack", "Moderate attack with moderate entropy -> Attack alert"),
        ]

    @staticmethod
    def compute_decision_features(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute confidence, prob_normal, and normalized Shannon entropy from ANN probability distribution.

        probs: shape (N, 5) with class probabilities summing to 1.0.
        Returns: (confidence, prob_normal, norm_entropy), each of shape (N,).
        """
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        confidence = np.max(probs, axis=1)
        prob_normal = probs[:, 0]

        # Normalized Shannon entropy: -sum(p * ln(p)) / ln(5)
        num_classes = probs.shape[1]
        entropy = -np.sum(probs * np.log(probs), axis=1)
        max_entropy = np.log(num_classes)
        norm_entropy = np.clip(entropy / max_entropy, 0.0, 1.0)

        return confidence, prob_normal, norm_entropy

    def evaluate_severity(self, probs: np.ndarray) -> np.ndarray:
        """Vectorized Mamdani fuzzy evaluation producing attack severity score in [0.0, 1.0]."""
        confidence, prob_normal, norm_entropy = self.compute_decision_features(probs)
        n = len(confidence)

        # Fuzzify Confidence
        mu_conf_low = np.maximum(0.0, np.minimum(1.0, (0.50 - confidence) / 0.25))
        mu_conf_med = np.maximum(
            0.0,
            np.minimum((confidence - 0.35) / 0.15, (0.80 - confidence) / 0.15),
        )
        mu_conf_high = np.maximum(0.0, np.minimum(1.0, (confidence - 0.65) / 0.25))

        # Fuzzify Probability of Normal
        mu_norm_low = np.maximum(0.0, np.minimum(1.0, (0.40 - prob_normal) / 0.30))
        mu_norm_med = np.maximum(
            0.0,
            np.minimum((prob_normal - 0.25) / 0.25, (0.75 - prob_normal) / 0.25),
        )
        mu_norm_high = np.maximum(0.0, np.minimum(1.0, (prob_normal - 0.60) / 0.30))

        # Fuzzify Prediction Entropy
        mu_ent_low = np.maximum(0.0, np.minimum(1.0, (0.35 - norm_entropy) / 0.25))
        mu_ent_med = np.maximum(
            0.0,
            np.minimum((norm_entropy - 0.25) / 0.25, (0.75 - norm_entropy) / 0.25),
        )
        mu_ent_high = np.maximum(0.0, np.minimum(1.0, (norm_entropy - 0.55) / 0.25))

        conf_lookup = {"Low": mu_conf_low, "Medium": mu_conf_med, "High": mu_conf_high, None: np.ones(n)}
        norm_lookup = {"Low": mu_norm_low, "Medium": mu_norm_med, "High": mu_norm_high, None: np.ones(n)}
        ent_lookup = {"Low": mu_ent_low, "Medium": mu_ent_med, "High": mu_ent_high, None: np.ones(n)}

        total_weight = np.zeros(n, dtype=np.float64)
        weighted_severity = np.zeros(n, dtype=np.float64)

        for c_t, n_t, e_t, sev_t, _ in self.rules:
            # Antecedent evaluation using min t-norm
            w = np.minimum(conf_lookup[c_t], np.minimum(norm_lookup[n_t], ent_lookup[e_t]))
            sev_center = self.severity_centers[sev_t]
            total_weight += w
            weighted_severity += w * sev_center

        # Safe defuzzification with fallback to (1.0 - prob_normal)
        nonzero_mask = total_weight > 1e-7
        severity = np.empty(n, dtype=np.float64)
        severity[nonzero_mask] = weighted_severity[nonzero_mask] / total_weight[nonzero_mask]
        severity[~nonzero_mask] = 1.0 - prob_normal[~nonzero_mask]

        return np.clip(severity, 0.0, 1.0).astype(np.float32)

    def predict(
        self,
        probs: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute hard predictions, fuzzy severity, 3-level alert levels, and fuzzy-adjusted predictions.

        Returns dictionary containing:
        - hard_preds: np.ndarray (standard argmax)
        - fuzzy_preds: np.ndarray (adjusted for false positives and rare class sensitivity)
        - severity: np.ndarray of float in [0, 1]
        - alert_indices: np.ndarray (0=Normal, 1=Suspicious, 2=Attack)
        - alert_names: List[str] of strings
        - confidence: np.ndarray
        - prob_normal: np.ndarray
        - norm_entropy: np.ndarray
        """
        probs = np.asarray(probs, dtype=np.float64)
        hard_preds = np.argmax(probs, axis=1).astype(np.int32)
        confidence, prob_normal, norm_entropy = self.compute_decision_features(probs)
        severity = self.evaluate_severity(probs)

        low_cutoff, high_cutoff = self.suspicious_band

        # Assign 3-level alert: 0=Normal, 1=Suspicious, 2=Attack
        alert_indices = np.zeros(len(severity), dtype=np.int32)
        suspicious_mask = (severity >= low_cutoff) & (severity <= high_cutoff)
        attack_mask = severity > high_cutoff

        alert_indices[suspicious_mask] = 1
        alert_indices[attack_mask] = 2

        # Fuzzy adjusted 5-class predictions
        fuzzy_preds = hard_preds.copy()

        # 1. False positive suppression:
        # If classified as an attack by argmax but alert is Normal (severity < low_cutoff)
        # and attack confidence is weak (< 0.55), adjust to Normal (0).
        fp_mitigation_mask = (hard_preds != 0) & (alert_indices == 0) & (confidence < 0.55)
        fuzzy_preds[fp_mitigation_mask] = 0

        # 2. Rare-class false negative recovery:
        # If classified as Normal by argmax but alert is Attack (severity > high_cutoff)
        # and there is non-trivial attack probability (> 0.25), assign to highest attack class.
        attack_subprobs = probs[:, 1:]  # Attack classes 1..4
        best_attack_class = np.argmax(attack_subprobs, axis=1) + 1
        best_attack_prob = np.max(attack_subprobs, axis=1)
        fn_recovery_mask = (hard_preds == 0) & (alert_indices == 2) & (best_attack_prob > 0.25)
        fuzzy_preds[fn_recovery_mask] = best_attack_class[fn_recovery_mask]

        alert_names = [self.alert_levels[idx] for idx in alert_indices]

        return {
            "hard_preds": hard_preds,
            "fuzzy_preds": fuzzy_preds,
            "severity": severity,
            "alert_indices": alert_indices,
            "alert_names": alert_names,
            "confidence": confidence.astype(np.float32),
            "prob_normal": prob_normal.astype(np.float32),
            "norm_entropy": norm_entropy.astype(np.float32),
        }

    def compute_alert_metrics(
        self,
        alert_indices: np.ndarray,
        y_true: np.ndarray,
        raw_attacks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute alert-level metrics treating Suspicious (1) and Attack (2) as 'flagged'.

        Metrics:
        1. attack_recall_flagged: Attack recall including Suspicious (P(flagged|attack))
        2. attack_alert_precision: Precision of Attack (2) alerts (P(attack|alert==Attack))
        3. normal_flagged_rate: % of Normal flagged (false alarms) (P(flagged|normal))
        4. analyst_workload_rate: % of records flagged (analyst workload) (P(flagged))
        5. per_attack_flagged_fraction: Dict mapping attack name to fraction flagged.
        """
        alert_indices = np.asarray(alert_indices, dtype=np.int32)
        y_true = np.asarray(y_true, dtype=np.int32)
        n_total = len(y_true)

        is_attack = (y_true != 0)
        is_normal = (y_true == 0)
        flagged = (alert_indices >= 1)
        attack_alert = (alert_indices == 2)

        n_attack = int(np.sum(is_attack))
        n_normal = int(np.sum(is_normal))
        n_attack_alerts = int(np.sum(attack_alert))
        n_flagged = int(np.sum(flagged))

        attack_recall_flagged = float(np.sum(is_attack & flagged) / max(1, n_attack)) if n_attack > 0 else 0.0
        attack_alert_precision = float(np.sum(is_attack & attack_alert) / max(1, n_attack_alerts)) if n_attack_alerts > 0 else 0.0
        normal_flagged_rate = float(np.sum(is_normal & flagged) / max(1, n_normal)) if n_normal > 0 else 0.0
        analyst_workload_rate = float(n_flagged / max(1, n_total))

        per_attack: Dict[str, float] = {}
        if raw_attacks is not None:
            raw_arr = np.asarray(raw_attacks, dtype=str)
            unique_attacks = sorted(list(set(raw_arr[raw_arr != "normal"])))
            for att in unique_attacks:
                mask = (raw_arr == att)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    per_attack[str(att)] = round(float(np.sum(mask & flagged) / cnt), 4)

        return {
            "attack_recall_flagged": round(attack_recall_flagged, 5),
            "attack_alert_precision": round(attack_alert_precision, 5),
            "normal_flagged_rate": round(normal_flagged_rate, 5),
            "analyst_workload_rate": round(analyst_workload_rate, 5),
            "total_flagged_count": n_flagged,
            "total_records": n_total,
            "per_attack_flagged_fraction": per_attack,
        }

    def compute_suspicious_statistics(
        self,
        alert_indices: np.ndarray,
        true_labels: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute breakdown of records placed into the Suspicious band."""
        if classes is None:
            classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]

        suspicious_mask = alert_indices == 1
        total_records = len(alert_indices)
        suspicious_count = int(np.sum(suspicious_mask))
        percentage = round((suspicious_count / max(1, total_records)) * 100.0, 2)

        if suspicious_count > 0:
            suspicious_true = true_labels[suspicious_mask]
            comp = {}
            for cls_idx, cls_name in enumerate(classes):
                cnt = int(np.sum(suspicious_true == cls_idx))
                pct = round((cnt / suspicious_count) * 100.0, 2)
                comp[cls_name] = {"count": cnt, "percentage": pct}
        else:
            comp = {cls_name: {"count": 0, "percentage": 0.0} for cls_name in classes}

        return {
            "total_records": total_records,
            "suspicious_count": suspicious_count,
            "suspicious_percentage": percentage,
            "composition": comp,
        }

    def tune_on_validation(
        self,
        val_probs: np.ndarray,
        y_val: np.ndarray,
        false_alarm_penalty: float = 0.30,
        workload_penalty: float = 0.10,
        beta_fpr_penalty: Optional[float] = None,
        classes: Optional[List[str]] = None,
        raw_attacks_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Optimize suspicious_band thresholds [low_cutoff, high_cutoff] strictly on validation split.

        Objective: Maximize flagged-attack recall minus workload and false-alarm penalties:
        Score = attack_recall_flagged - false_alarm_penalty * normal_flagged_rate - workload_penalty * analyst_workload_rate.
        NEVER accepts or evaluates on test set data.
        """
        from sklearn.metrics import f1_score

        if beta_fpr_penalty is not None and false_alarm_penalty == 0.30:
            false_alarm_penalty = beta_fpr_penalty

        logger.info("Tuning Fuzzy Decision Layer suspicious band strictly on validation split (flagged recall vs false-alarm/workload penalties)...")
        val_probs = np.asarray(val_probs, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.int32)

        # Baseline performance before fuzzy decision adjustment
        hard_preds = np.argmax(val_probs, axis=1)
        base_f1 = float(f1_score(y_val, hard_preds, average="macro", zero_division=0))
        normal_mask = (y_val == 0)
        base_fpr = float(np.sum(normal_mask & (hard_preds != 0)) / max(1, np.sum(normal_mask)))

        # Baseline alert metrics with default band
        init_res = self.predict(val_probs)
        init_alert_metrics = self.compute_alert_metrics(init_res["alert_indices"], y_val, raw_attacks=raw_attacks_val)
        base_score = (
            init_alert_metrics["attack_recall_flagged"]
            - false_alarm_penalty * init_alert_metrics["normal_flagged_rate"]
            - workload_penalty * init_alert_metrics["analyst_workload_rate"]
        )

        best_score = -float("inf")
        best_band = self.suspicious_band
        best_metrics = init_alert_metrics

        # Grid search over candidate cutoff intervals
        low_candidates = [0.25, 0.30, 0.35, 0.40, 0.45]
        high_candidates = [0.55, 0.60, 0.65, 0.70, 0.75]

        for low in low_candidates:
            for high in high_candidates:
                if high - low < 0.15:
                    continue
                self.suspicious_band = [low, high]
                res = self.predict(val_probs)
                m = self.compute_alert_metrics(res["alert_indices"], y_val, raw_attacks=raw_attacks_val)
                score = (
                    m["attack_recall_flagged"]
                    - false_alarm_penalty * m["normal_flagged_rate"]
                    - workload_penalty * m["analyst_workload_rate"]
                )

                if score > best_score:
                    best_score = score
                    best_band = [low, high]
                    best_metrics = m

        self.suspicious_band = best_band
        self.is_tuned = True

        # Generate final validation statistics
        final_res = self.predict(val_probs)
        final_alert_metrics = self.compute_alert_metrics(final_res["alert_indices"], y_val, raw_attacks=raw_attacks_val)
        susp_stats = self.compute_suspicious_statistics(
            final_res["alert_indices"], y_val, classes=classes
        )

        stats = {
            "validation_baseline": {
                "macro_f1": round(base_f1, 5),
                "fpr": round(base_fpr, 5),
                "objective_score": round(base_score, 5),
            },
            "validation_fuzzy_adjusted": {
                "macro_f1": round(base_f1, 5),
                "fpr": round(base_fpr, 5),
                "objective_score": round(best_score, 5),
                "f1_improvement": 0.0,
                "fpr_reduction": 0.0,
            },
            "tuning_objective_score": round(best_score, 5),
            "validation_alert_metrics": final_alert_metrics,
            "tuned_suspicious_band": best_band,
            "suspicious_band_validation_stats": susp_stats,
        }

        self.tuning_stats = stats
        logger.info(
            f"Tuned Suspicious Band: {best_band} (Objective: {best_score:.4f} | "
            f"Flagged Recall: {final_alert_metrics['attack_recall_flagged']:.4f} | "
            f"False Alarms: {final_alert_metrics['normal_flagged_rate']:.4f} | "
            f"Workload: {final_alert_metrics['analyst_workload_rate']:.4f})"
        )
        return stats

    def export_rules_dict(self) -> List[Dict[str, Any]]:
        """Export interpretable decision rules definition."""
        rule_list = []
        for i, (c_t, n_t, e_t, s_t, desc) in enumerate(self.rules, start=1):
            ante_parts = []
            if c_t:
                ante_parts.append(f"Confidence is {c_t}")
            if n_t:
                ante_parts.append(f"P(Normal) is {n_t}")
            if e_t:
                ante_parts.append(f"Entropy is {e_t}")
            antecedent = "IF " + " AND ".join(ante_parts)
            consequent = f"THEN Severity is {s_t}"
            rule_list.append({
                "rule_id": i,
                "antecedent": antecedent,
                "consequent": consequent,
                "description": desc,
            })
        return rule_list


def export_all_fuzzy_rules(output_path: Path) -> Dict[str, Any]:
    """Export both GA adaptive rules and decision layer rules to a JSON file."""
    ga_controller = FuzzyGAController()
    decision_layer = FuzzyDecisionLayer()
    rules_dict = {
        "mamdani_ga_controller": {
            "num_rules": len(ga_controller.rules),
            "inputs": {
                "diversity": "Normalized mean pairwise Hamming distance [0, 1]",
                "stagnation": "Normalized generations without fitness improvement [0, 1]",
            },
            "outputs": {
                "mutation_rate": f"Adapted mutation rate [{ga_controller.min_pm}, {ga_controller.max_pm}]",
                "crossover_rate": f"Adapted crossover rate [{ga_controller.min_pc}, {ga_controller.max_pc}]",
            },
            "rules": ga_controller.export_rules_dict(),
        },
        "fuzzy_decision_layer": {
            "num_rules": len(decision_layer.rules),
            "inputs": {
                "confidence": "Max class probability [0.2, 1.0]",
                "prob_normal": "P(Normal) probability [0.0, 1.0]",
                "prediction_entropy": "Normalized Shannon entropy [0.0, 1.0]",
            },
            "outputs": {
                "severity": "Continuous attack severity score [0.0, 1.0]",
                "alert_level": "3-level categorical alert: Normal / Suspicious / Attack",
            },
            "rules": decision_layer.export_rules_dict(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rules_dict, output_path)
    logger.info(f"Exported combined fuzzy rules specification to {output_path}")
    return rules_dict


