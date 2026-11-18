"""Kaggle ingestion and strict schema validation for Bank Customer Churn data."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.core.churn_config import ChurnPathConfig, FeaturePartitionConfig
from src.core.churn_contracts import ChurnDatasetBundle

KAGGLE_DATASET_SLUG = "radheshyamkollipara/bank-customer-churn"


def resolve_kaggle_executable() -> str:
    """Locate the Kaggle CLI executable in active Python Scripts or PATH.

    Returns:
        Filesystem command path to ``kaggle.exe`` or ``kaggle``.

    Raises:
        FileNotFoundError: If no Kaggle CLI executable is discoverable.
    """
    sibling_scripts = Path(sys.executable).resolve().parent / "Scripts" / "kaggle.exe"
    if sibling_scripts.exists():
        return str(sibling_scripts)
    discovered = shutil.which("kaggle")
    if discovered:
        return discovered
    raise FileNotFoundError("Kaggle CLI executable not found in Python Scripts or PATH.")


def download_churn_dataset(path_config: ChurnPathConfig) -> Path:
    """Download ``Customer-Churn-Records.csv`` from Kaggle if not yet cached locally.

    Inputs:
        path_config: Immutable path configuration specifying raw CSV destination and
            ``kaggle_config_dir`` containing ``kaggle.json``.

    Returns:
        Path to the downloaded raw CSV file.

    Edge Cases:
        Skips network download when the raw CSV already exists and is non-empty.
        Raises ``FileNotFoundError`` if ``kaggle.json`` is missing and ``RuntimeError``
        if the Kaggle CLI exits non-zero or fails to produce the expected CSV.
    """
    target_csv = path_config.raw_csv_path
    if target_csv.exists() and target_csv.stat().st_size > 0:
        return target_csv
    kaggle_json = path_config.kaggle_config_dir / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(f"Missing kaggle.json at {kaggle_json}")

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(path_config.kaggle_config_dir)
    cmd = [
        resolve_kaggle_executable(),
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET_SLUG,
        "-p",
        str(target_csv.parent),
        "--unzip",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr or result.stdout}")
    if not target_csv.exists() or target_csv.stat().st_size == 0:
        raise RuntimeError(f"Expected dataset file missing after unzip: {target_csv}")
    return target_csv


def has_valid_churn_schema(
    df: pd.DataFrame,
    partition_config: FeaturePartitionConfig,
) -> bool:
    """Verify structural invariants of the raw 18-column bank churn DataFrame.

    Inputs:
        df: Candidate DataFrame loaded from CSV.
        partition_config: Expected column definitions and target name.

    Returns:
        ``True`` if non-empty, all 18 columns exist, zero nulls appear in required
        columns, ``CustomerId`` is unique, and ``Exited`` is strictly binary ``{0, 1}``.
    """
    if df.empty:
        return False
    expected = set(partition_config.expected_raw_columns())
    if not expected.issubset(set(df.columns)):
        return False
    if df[list(expected)].isna().any().any():
        return False
    if df["CustomerId"].duplicated().any():
        return False
    target_values = set(df[partition_config.target_column].unique().tolist())
    return target_values.issubset({0, 1}) and len(target_values) == 2


def load_and_validate_churn_csv(
    csv_path: Path,
    partition_config: FeaturePartitionConfig,
) -> ChurnDatasetBundle:
    """Read raw bank churn CSV, validate schema, and build an immutable bundle.

    Inputs:
        csv_path: Path to ``Customer-Churn-Records.csv``.
        partition_config: Schema and column role specification.

    Returns:
        Frozen ``ChurnDatasetBundle`` containing a defensive copy of the DataFrame.

    Raises:
        FileNotFoundError: When ``csv_path`` does not exist.
        ValueError: When schema, uniqueness, or target domain validation fails.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Churn CSV not found at {csv_path}")
    raw_df = pd.read_csv(csv_path)
    if not has_valid_churn_schema(raw_df, partition_config):
        raise ValueError("Churn CSV failed schema, null, uniqueness, or binary target validation.")
    churn_rate = float(raw_df[partition_config.target_column].mean())
    zero_balance_rate = float((raw_df["Balance"] == 0.0).mean())
    return ChurnDatasetBundle(
        raw_df=raw_df.copy(),
        n_rows=int(len(raw_df)),
        n_cols=int(raw_df.shape[1]),
        churn_rate=churn_rate,
        zero_balance_rate=zero_balance_rate,
    )
