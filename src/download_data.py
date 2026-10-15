"""Dataset acquisition and validation module for NSL-KDD.

Adheres strictly to credential security and dataset validation specifications.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("download_data")

EXPECTED_COLUMNS = 43
EXPECTED_TRAIN_ROWS_APPROX = 125973
EXPECTED_TEST_ROWS_APPROX = 22544
ROW_TOLERANCE_PCT = 0.05  # Within 5% of expected row count


class DatasetBlockedError(Exception):
    """Raised when dataset acquisition is blocked due to missing credentials or network issues."""
    pass


class DatasetValidationError(Exception):
    """Raised when downloaded data fails integrity or schema checks."""
    pass


def check_kaggle_credentials() -> bool:
    """Check for Kaggle credentials without logging, printing, or exposing their contents."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file() and kaggle_json.stat().st_size > 0:
        return True

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return True

    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token:
        return True

    return False


def validate_file(file_path: Path, expected_name: str, approx_rows: int) -> Dict[str, any]:
    """Validate that an NSL-KDD file satisfies the 43-column schema, row bounds, and numeric difficulty."""
    if not file_path.exists():
        raise DatasetValidationError(f"Expected file does not exist: {file_path}")

    # Read with no header using pandas
    try:
        df_head = pd.read_csv(file_path, header=None, nrows=100)
    except Exception as e:
        raise DatasetValidationError(f"Failed to read file {file_path.name}: {e}")

    num_cols = df_head.shape[1]
    if num_cols != EXPECTED_COLUMNS:
        raise DatasetValidationError(
            f"File {file_path.name} has {num_cols} columns, expected {EXPECTED_COLUMNS}."
        )

    # Check last column is numeric difficulty
    last_col = df_head.iloc[:, -1]
    if not pd.api.types.is_numeric_dtype(last_col):
        # Attempt to convert to numeric to verify
        converted = pd.to_numeric(last_col, errors="coerce")
        if converted.isna().any():
            raise DatasetValidationError(
                f"File {file_path.name} last column (difficulty) is not numeric."
            )

    # Count total rows
    total_rows = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total_rows += 1

    lower_bound = int(approx_rows * (1.0 - ROW_TOLERANCE_PCT))
    upper_bound = int(approx_rows * (1.0 + ROW_TOLERANCE_PCT))

    if not (lower_bound <= total_rows <= upper_bound):
        raise DatasetValidationError(
            f"File {file_path.name} row count {total_rows} is not plausible. "
            f"Expected approx {approx_rows} (between {lower_bound} and {upper_bound})."
        )

    return {
        "file": file_path.name,
        "rows": total_rows,
        "columns": num_cols,
        "valid": True,
    }


def locate_and_relocate_files(raw_dir: Path) -> Tuple[Path, Path]:
    """Locate KDDTrain+.txt and KDDTest+.txt in raw_dir or any subdirectory and move to raw_dir."""
    target_train = raw_dir / "KDDTrain+.txt"
    target_test = raw_dir / "KDDTest+.txt"

    if target_train.exists() and target_test.exists():
        return target_train, target_test

    # Search recursively for matching files
    found_train = None
    found_test = None

    for p in raw_dir.rglob("*"):
        if p.is_file():
            name_lower = p.name.lower()
            if "kddtrain+" in name_lower and p.suffix.lower() == ".txt":
                found_train = p
            elif "kddtest+" in name_lower and p.suffix.lower() == ".txt":
                found_test = p

    if found_train and found_train != target_train:
        logger.info(f"Relocating {found_train} -> {target_train}")
        shutil.copy2(found_train, target_train)

    if found_test and found_test != target_test:
        logger.info(f"Relocating {found_test} -> {target_test}")
        shutil.copy2(found_test, target_test)

    return target_train, target_test


def download_via_kaggle_cli(raw_dir: Path) -> bool:
    """Attempt download using the kaggle CLI."""
    logger.info("Attempting download via Kaggle CLI: kaggle datasets download -d hassan06/nslkdd -p data/raw --unzip")
    try:
        # Use python -m kaggle to ensure using the active python/venv
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "hassan06/nslkdd",
            "-p",
            str(raw_dir),
            "--unzip",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            logger.info("Kaggle CLI download completed successfully.")
            return True
        else:
            logger.warning(f"Kaggle CLI download exited with code {result.returncode}. Stderr: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"Kaggle CLI download failed: {e}")
        return False


def download_via_kagglehub(raw_dir: Path) -> bool:
    """Attempt download using kagglehub package."""
    logger.info("Attempting download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("hassan06/nslkdd")
        downloaded_path = Path(path)
        logger.info(f"Kagglehub downloaded to {downloaded_path}. Copying files to {raw_dir}...")
        for item in downloaded_path.rglob("*"):
            if item.is_file():
                dest = raw_dir / item.name
                shutil.copy2(item, dest)
        return True
    except Exception as e:
        logger.warning(f"Kagglehub download failed: {e}")
        return False


def ensure_nsl_kdd_dataset(raw_dir: Path = Path("data/raw")) -> Tuple[Path, Path]:
    """Ensure NSL-KDD dataset is present and valid in data/raw/.

    Follows the exact 7-step priority order specified.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_train = raw_dir / "KDDTrain+.txt"
    target_test = raw_dir / "KDDTest+.txt"

    # Step 1: Check if already present and valid
    if target_train.exists() and target_test.exists():
        logger.info(f"Existing dataset files found in {raw_dir}. Validating...")
        try:
            val_train = validate_file(target_train, "KDDTrain+.txt", EXPECTED_TRAIN_ROWS_APPROX)
            val_test = validate_file(target_test, "KDDTest+.txt", EXPECTED_TEST_ROWS_APPROX)
            logger.info(f"KDDTrain+.txt validated ({val_train['rows']} rows, {val_train['columns']} columns).")
            logger.info(f"KDDTest+.txt validated ({val_test['rows']} rows, {val_test['columns']} columns).")
            return target_train, target_test
        except DatasetValidationError as e:
            logger.warning(f"Existing files failed validation: {e}. Re-checking acquisition options.")

    # Step 2: Check for Kaggle credentials WITHOUT printing them
    has_creds = check_kaggle_credentials()
    if not has_creds:
        logger.error("No Kaggle credentials detected in ~/.kaggle/kaggle.json or environment variables.")
        raise DatasetBlockedError(
            "Kaggle credentials are missing or unconfigured.\n"
            "Action required:\n"
            "1. Download the NSL-KDD dataset manually from: https://www.kaggle.com/datasets/hassan06/nslkdd\n"
            "2. Extract KDDTrain+.txt and KDDTest+.txt into data/raw/ in this project root.\n"
            "3. Reply 'continue'."
        )

    # Step 3: Try Kaggle CLI, fallback to kagglehub
    download_success = download_via_kaggle_cli(raw_dir)
    if not download_success:
        download_success = download_via_kagglehub(raw_dir)

    if not download_success:
        raise DatasetBlockedError(
            "Kaggle download failed (authentication, terms not accepted, quota, or network blocked).\n"
            "Action required:\n"
            "1. Download the NSL-KDD dataset manually from: https://www.kaggle.com/datasets/hassan06/nslkdd\n"
            "2. Extract KDDTrain+.txt and KDDTest+.txt into data/raw/ in this project root.\n"
            "3. Reply 'continue'."
        )

    # Step 4: Locate and relocate files
    target_train, target_test = locate_and_relocate_files(raw_dir)
    if not (target_train.exists() and target_test.exists()):
        raise DatasetBlockedError(
            "Download completed but KDDTrain+.txt or KDDTest+.txt was not found in the extracted files.\n"
            "Action required:\n"
            "Extract KDDTrain+.txt and KDDTest+.txt into data/raw/ and reply 'continue'."
        )

    # Step 5: Validate
    val_train = validate_file(target_train, "KDDTrain+.txt", EXPECTED_TRAIN_ROWS_APPROX)
    val_test = validate_file(target_test, "KDDTest+.txt", EXPECTED_TEST_ROWS_APPROX)
    logger.info(f"Dataset successfully acquired and validated: Train={val_train['rows']} rows, Test={val_test['rows']} rows.")

    return target_train, target_test


if __name__ == "__main__":
    try:
        train_p, test_p = ensure_nsl_kdd_dataset()
        print(f"Dataset ready:\nTrain: {train_p}\nTest: {test_p}")
    except DatasetBlockedError as dbe:
        print(f"BLOCKED: {dbe}", file=sys.stderr)
        sys.exit(2)
    except DatasetValidationError as dve:
        print(f"VALIDATION FAILED: {dve}", file=sys.stderr)
        sys.exit(3)
