"""Data loading module for NSL-KDD dataset."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.config import AppConfig, load_config
from src.download_data import ensure_nsl_kdd_dataset
from src.utils import setup_logger

logger = setup_logger("data_loader")


def get_attack_to_category_mapping(attack_mapping_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert attack_mapping dictionary to map attack name -> 5-class category."""
    mapping = {}
    for category, attacks in attack_mapping_dict.items():
        for attack in attacks:
            mapping[attack.lower().strip()] = category
    return mapping


def load_raw_nsl_kdd(
    config: Optional[AppConfig] = None,
    raw_dir: Path = Path("data/raw"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load KDDTrain+ and KDDTest+ raw files, validate schemas, and map attack labels to 5 classes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) with 41 features + 'attack_category' + 'raw_label'.
    """
    if config is None:
        config = load_config()

    # Step 1: Ensure raw files exist and pass validation
    train_path, test_path = ensure_nsl_kdd_dataset(raw_dir=raw_dir)

    column_names = config.data.column_names
    target_col = config.data.target_column
    diff_col = config.data.difficulty_column
    attack_map = get_attack_to_category_mapping(config.data.attack_mapping)

    logger.info(f"Loading raw train dataset from {train_path}...")
    train_df = pd.read_csv(train_path, header=None, names=column_names)
    logger.info(f"Raw train shape: {train_df.shape}")

    logger.info(f"Loading raw test dataset from {test_path}...")
    test_df = pd.read_csv(test_path, header=None, names=column_names)
    logger.info(f"Raw test shape: {test_df.shape}")

    # Map attack labels
    for name, df in [("Train", train_df), ("Test", test_df)]:
        # Strip any whitespace
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        
        # Check for unmapped attack names
        unique_labels = set(df[target_col].unique())
        unmapped = unique_labels - set(attack_map.keys())
        if unmapped:
            logger.error(f"{name} set contains unmapped labels: {unmapped}")
            raise ValueError(
                f"{name} set contains unmapped labels: {unmapped}. Cannot proceed without explicit mapping."
            )

        df["attack_category"] = df[target_col].map(attack_map)
        df["raw_label"] = df[target_col]

        # Verify no nulls in target
        if df["attack_category"].isna().any():
            raise ValueError(f"Null values produced while mapping attack categories in {name} set.")

    logger.info("Raw datasets successfully loaded and attack labels mapped.")
    return train_df, test_df
