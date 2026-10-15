"""Preprocessing and feature engineering module for NSL-KDD."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import AppConfig, load_config
from src.data_loader import load_raw_nsl_kdd
from src.utils import load_artifact, save_artifact, save_json, setup_logger

logger = setup_logger("preprocessing")


class NSLKDDPreprocessor:
    """Preprocessor for NSL-KDD dataset.

    Enforces strict data hygiene:
    - Never uses test data for fitting encoders/scalers.
    - Excludes label and difficulty columns from feature matrix.
    - Maintains exact mapping between 41 original features and one-hot encoded columns.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        classes: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        difficulty_column: str = "difficulty",
        target_column: str = "label",
    ):
        self.scaler_type = scaler_type
        self.classes = classes or ["Normal", "DoS", "Probe", "R2L", "U2R"]
        self.categorical_features = categorical_features or ["protocol_type", "service", "flag"]
        self.difficulty_column = difficulty_column
        self.target_column = target_column

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.original_feature_names: List[str] = []
        self.numeric_features: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.feature_to_encoded_indices: Dict[int, List[int]] = {}
        self.feature_name_to_encoded_indices: Dict[str, List[int]] = {}
        self.feature_defaults: Dict[str, Any] = {}
        self.fuzzy_input_layer: Optional[Any] = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, fuzzy_config: Optional[Any] = None) -> "NSLKDDPreprocessor":
        """Fit scaler, encoder, fuzzy partition models, and build feature column index mappings using training data only."""
        from src.fuzzy import FuzzyInputLayer

        exclude_cols = {self.target_column, self.difficulty_column, "attack_category", "raw_label"}
        self.original_feature_names = [col for col in train_df.columns if col not in exclude_cols]

        if len(self.original_feature_names) != 41:
            raise ValueError(
                f"Expected 41 original input features, found {len(self.original_feature_names)}: {self.original_feature_names}"
            )

        self.numeric_features = [col for col in self.original_feature_names if col not in self.categorical_features]

        # Fit numeric scaler
        self.scaler.fit(train_df[self.numeric_features].astype(np.float32))

        # Fit categorical one-hot encoder
        self.encoder.fit(train_df[self.categorical_features].astype(str))

        # Store sensible default values (medians for numeric, modes for categorical) for single-record prediction UI
        for col in self.numeric_features:
            self.feature_defaults[col] = float(train_df[col].median())
        for col in self.categorical_features:
            self.feature_defaults[col] = str(train_df[col].mode().iloc[0])

        # Get encoded column names
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.categorical_features))
        self.encoded_feature_names = self.numeric_features + encoded_cat_names

        # Build chromosome mapping: each original feature (0 to 40) -> list of indices in encoded matrix
        self.feature_to_encoded_indices = {}
        self.feature_name_to_encoded_indices = {}

        num_indices_map = {feat: idx for idx, feat in enumerate(self.numeric_features)}
        num_len = len(self.numeric_features)

        cat_feature_indices: Dict[str, List[int]] = {col: [] for col in self.categorical_features}
        for offset, encoded_name in enumerate(encoded_cat_names):
            # encoded_name is e.g. "protocol_type_tcp"
            for cat_col in self.categorical_features:
                if encoded_name.startswith(cat_col + "_"):
                    cat_feature_indices[cat_col].append(num_len + offset)
                    break

        for orig_idx, feat_name in enumerate(self.original_feature_names):
            if feat_name in self.numeric_features:
                indices = [num_indices_map[feat_name]]
            elif feat_name in self.categorical_features:
                indices = cat_feature_indices[feat_name]
            else:
                raise ValueError(f"Feature {feat_name} not recognized as numeric or categorical.")

            self.feature_to_encoded_indices[orig_idx] = indices
            self.feature_name_to_encoded_indices[feat_name] = indices

        # Validate that all encoded columns are covered exactly once
        all_mapped_indices = [idx for sublist in self.feature_to_encoded_indices.values() for idx in sublist]
        if sorted(all_mapped_indices) != list(range(len(self.encoded_feature_names))):
            raise ValueError("Feature mapping does not perfectly partition all encoded feature columns.")

        # Fit fuzzy Gaussian membership input layer on training numeric features
        self.fuzzy_input_layer = FuzzyInputLayer(config=fuzzy_config)
        self.fuzzy_input_layer.fit(
            train_df[self.numeric_features].to_numpy(dtype=np.float32),
            self.numeric_features,
        )

        self.is_fitted = True
        logger.info(
            f"Preprocessor fitted successfully: {len(self.original_feature_names)} original features -> {len(self.encoded_feature_names)} encoded columns + Fuzzy Input Layer."
        )
        return self

    def transform_neuro_fuzzy(
        self,
        X_encoded_full: np.ndarray,
        selected_chromosome: np.ndarray,
        input_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Apply neuro-fuzzy transformation for selected chromosome."""
        if not self.is_fitted or self.fuzzy_input_layer is None:
            raise RuntimeError("Preprocessor must be fitted with fuzzy layer before transform_neuro_fuzzy.")
        return self.fuzzy_input_layer.transform_selected_features(
            X_encoded_full=X_encoded_full,
            preprocessor=self,
            selected_chromosome=selected_chromosome,
            input_mode=input_mode,
        )


    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw dataframe features into scaled, one-hot encoded numpy array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming features.")

        # Ensure all required features are present
        missing = [f for f in self.original_feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing required features: {missing}")

        num_scaled = self.scaler.transform(df[self.numeric_features].astype(np.float32))
        cat_encoded = self.encoder.transform(df[self.categorical_features].astype(str))

        return np.hstack([num_scaled, cat_encoded]).astype(np.float32)

    def transform_labels(self, labels: pd.Series) -> np.ndarray:
        """Transform attack_category series into integer classes (0 to 4)."""
        labels_clean = labels.astype(str).str.strip()
        encoded = labels_clean.map(self.class_to_idx)
        if encoded.isna().any():
            unmapped = labels_clean[encoded.isna()].unique()
            raise ValueError(f"Unrecognized class labels during label transform: {unmapped}")
        return encoded.to_numpy(dtype=np.int32)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform features and labels (if present)."""
        X = self.transform_features(df)
        y = None
        if "attack_category" in df.columns:
            y = self.transform_labels(df["attack_category"])
        return X, y

    def get_column_indices_for_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """Translate a 41-bit binary chromosome into column indices for the encoded feature matrix."""
        if len(chromosome) != len(self.original_feature_names):
            raise ValueError(
                f"Chromosome length {len(chromosome)} does not match original features {len(self.original_feature_names)}"
            )

        indices: List[int] = []
        for gene_idx, is_active in enumerate(chromosome):
            if is_active:
                indices.extend(self.feature_to_encoded_indices[gene_idx])

        if not indices:
            raise ValueError("Chromosome selected zero features; at least one feature must be selected.")

        return np.array(sorted(indices), dtype=np.int32)


def process_dataset_and_generate_eda(
    config: Optional[AppConfig] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], NSLKDDPreprocessor]:
    """Execute complete Phase 1 pipeline: loading, train/val split, transformation, artifact saving, and EDA."""
    if config is None:
        config = load_config(mode="fast")

    # Step 1: Load raw data
    raw_train_df, raw_test_df = load_raw_nsl_kdd(config=config, raw_dir=config.paths.raw_train_path.parent)

    # Step 2: Stratified Train / Validation Split from KDDTrain+
    val_ratio = config.data.val_split_ratio
    train_split_df, val_split_df = train_test_split(
        raw_train_df,
        test_size=val_ratio,
        stratify=raw_train_df["attack_category"],
        random_state=seed,
    )

    logger.info(
        f"Data split: Train={train_split_df.shape[0]} rows, Val={val_split_df.shape[0]} rows, Test={raw_test_df.shape[0]} rows."
    )

    # Step 3: Fit Preprocessor on Train split only
    preprocessor = NSLKDDPreprocessor(
        scaler_type=config.data.scaler_type,
        classes=config.data.classes,
        categorical_features=config.data.categorical_features,
        difficulty_column=config.data.difficulty_column,
        target_column=config.data.target_column,
    )
    fuzzy_cfg = getattr(config, "fuzzy", None)
    in_layer_cfg = fuzzy_cfg.input_layer if fuzzy_cfg else None
    preprocessor.fit(train_split_df, fuzzy_config=in_layer_cfg)

    # Save fuzzy membership parameters if present
    if preprocessor.fuzzy_input_layer is not None:
        fuzzy_params_path = config.paths.results_dir / "fuzzy_membership_params.json"
        preprocessor.fuzzy_input_layer.save_params_json(fuzzy_params_path)

    # Transform all three splits
    X_train, y_train = preprocessor.transform(train_split_df)
    X_val, y_val = preprocessor.transform(val_split_df)
    X_test, y_test = preprocessor.transform(raw_test_df)

    # Step 4: Compute Class Weights on training set for handling imbalance
    unique_classes = np.arange(len(config.data.classes))
    computed_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, computed_weights)}

    # Step 5: Save Preprocessor Artifact
    preprocessor_path = config.paths.models_dir / "preprocessor.joblib"
    save_artifact(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor artifact to {preprocessor_path}")


    # Step 6: Generate Dataset Summary
    def compute_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        counts = df["attack_category"].value_counts().to_dict()
        total = len(df)
        pcts = {k: round((v / total) * 100, 2) for k, v in counts.items()}
        return {"total_samples": total, "class_counts": counts, "class_percentages": pcts}

    summary = {
        "dataset_name": "NSL-KDD",
        "split_ratio": {"train": 1.0 - val_ratio, "validation": val_ratio, "test": "Separate untouched KDDTest+"},
        "original_features_count": len(preprocessor.original_feature_names),
        "encoded_features_count": len(preprocessor.encoded_feature_names),
        "classes": config.data.classes,
        "class_weights_train": class_weights,
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "distributions": {
            "train": compute_distribution(train_split_df),
            "validation": compute_distribution(val_split_df),
            "test": compute_distribution(raw_test_df),
        },
        "categorical_features": config.data.categorical_features,
        "numeric_features_count": len(preprocessor.numeric_features),
    }

    summary_path = config.paths.results_dir / "dataset_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Saved dataset summary to {summary_path}")

    # Step 7: Generate Class Distribution EDA Plot
    config.paths.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(train_split_df, val_split_df, raw_test_df, config.paths.plots_dir / "class_distribution.png")

    dataset_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_weights": class_weights,
        "preprocessor": preprocessor,
        "raw_train": train_split_df,
        "raw_val": val_split_df,
        "raw_test": raw_test_df,
    }

    return dataset_dict, preprocessor


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate and save publication-quality class distribution plot comparing splits."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Prepare DataFrame
    dfs = []
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df["attack_category"].value_counts().reset_index()
        counts.columns = ["Attack Class", "Count"]
        counts["Split"] = split_name
        counts["Percentage"] = (counts["Count"] / len(df)) * 100
        dfs.append(counts)

    combined = pd.concat(dfs, ignore_index=True)
    palette = {"Train": "#1f77b4", "Val": "#ff7f0e", "Test": "#2ca02c"}

    # Plot raw counts (log scale for visibility of rare classes like U2R/R2L)
    sns.barplot(
        data=combined,
        x="Attack Class",
        y="Count",
        hue="Split",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Class Counts by Split (Log Scale)")
    axes[0].set_ylabel("Count (log)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # Plot relative percentages
    sns.barplot(
        data=combined,
        x="Attack Class",
        y="Percentage",
        hue="Split",
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title("Class Distribution Percentage by Split")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved class distribution plot to {output_path}")


if __name__ == "__main__":
    import src.preprocessing as prep_mod
    datasets, preprocessor = prep_mod.process_dataset_and_generate_eda()
    print("Preprocessing completed successfully.")
    print("X_train shape:", datasets["X_train"].shape)
    print("X_val shape:", datasets["X_val"].shape)
    print("X_test shape:", datasets["X_test"].shape)

