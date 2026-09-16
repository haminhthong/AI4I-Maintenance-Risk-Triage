"""Nạp, kiểm tra và chia dữ liệu AI4I theo snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contracts import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_CONTRACT,
    TARGET_COLUMN,
)
from .features import build_canonical_features, canonicalize_raw_dataframe
from .utils import LOGGER, save_json

DEFAULT_RAW_DATA_PATH = Path("data/raw/ai4i2020.csv")
DEFAULT_SPLIT_MANIFEST_PATH = Path("reports/split_manifest.json")
SPLIT_FRACTIONS = {"development": 0.70, "validation": 0.15, "test": 0.15}


def compute_dataset_sha256(path: str | Path) -> str:
    """Tính SHA256 ổn định giữa Windows và Linux bằng newline chuẩn LF."""
    csv_path = Path(path)
    if not csv_path.exists():
        return "file_not_found"
    content = csv_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_raw_dataset(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc CSV AI4I từ đường dẫn local."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại '{csv_path}'. Hãy chạy `python -m scripts.download_data`."
        )
    return pd.read_csv(csv_path)


def audit_dataset(
    df: pd.DataFrame,
    sha256_hash: str | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Kiểm tra schema, missing, duplicate và tỷ lệ failure trong bộ nhớ."""
    clean_df = canonicalize_raw_dataframe(df)
    if TARGET_COLUMN not in clean_df.columns:
        raise KeyError(f"Dataset thiếu cột target: {TARGET_COLUMN}")
    labels = clean_df[TARGET_COLUMN].astype(int)
    if not labels.isin({0, 1}).all():
        raise ValueError("machine_failure chỉ được chứa 0 hoặc 1.")

    numeric_ranges: dict[str, dict[str, float]] = {}
    for column in clean_df.select_dtypes(include=[np.number]).columns:
        values = clean_df[column].dropna()
        numeric_ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    report = {
        "dataset": "AI4I 2020",
        "observation_unit": "operating snapshot",
        "temporal_ordering": "unavailable",
        "rows": len(clean_df),
        "duplicate_rows": int(clean_df.duplicated().sum()),
        "missing_values": clean_df.isna().sum().loc[lambda values: values > 0].to_dict(),
        "raw_sha256": sha256_hash or "unknown",
        "target": {
            "column": TARGET_COLUMN,
            "failures": int(labels.sum()),
            "prevalence": float(labels.mean()),
        },
        "failure_mode_counts": {
            column: int(clean_df[column].astype(int).sum())
            for column in FAILURE_MODE_COLUMNS
            if column in clean_df.columns
        },
        "numeric_ranges": numeric_ranges,
        "excluded_from_features": [
            *IDENTIFIER_COLUMNS,
            TARGET_COLUMN,
            *FAILURE_MODE_COLUMNS,
        ],
    }
    if save_path is not None:
        save_json(save_path, report)
    return report


def _is_valid_split_manifest(
    manifest: dict[str, Any],
    total_rows: int,
    seed: int,
    dataset_sha256: str,
) -> bool:
    """Đảm bảo split thuộc đúng dataset, không trùng và bao phủ đủ dòng."""
    groups = [
        manifest.get("development_indices", []),
        manifest.get("validation_indices", []),
        manifest.get("test_indices", []),
    ]
    flattened = [index for group in groups for index in group]
    expected_counts = {
        "development": len(groups[0]),
        "validation": len(groups[1]),
        "test": len(groups[2]),
    }
    return (
        manifest.get("seed") == seed
        and manifest.get("dataset_sha256") == dataset_sha256
        and manifest.get("split_fractions") == SPLIT_FRACTIONS
        and manifest.get("split_counts") == expected_counts
        and len(flattened) == total_rows
        and len(set(flattened)) == total_rows
        and set(flattened) == set(range(total_rows))
    )


def create_or_load_split_manifest(
    df: pd.DataFrame,
    seed: int = 42,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    dataset_path: str | Path = DEFAULT_RAW_DATA_PATH,
) -> dict[str, Any]:
    """Tạo hoặc đọc split Development 70%, Validation 15%, Test 15%."""
    path = Path(manifest_path)
    dataset_sha256 = compute_dataset_sha256(dataset_path)
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if _is_valid_split_manifest(manifest, len(df), seed, dataset_sha256):
                return manifest
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Split manifest hiện tại không hợp lệ; sẽ tạo lại.")

    clean_df = canonicalize_raw_dataframe(df)
    labels = clean_df[TARGET_COLUMN].astype(int).to_numpy()
    indices = np.arange(len(clean_df))
    development_validation, test = train_test_split(
        indices,
        test_size=SPLIT_FRACTIONS["test"],
        stratify=labels,
        random_state=seed,
    )
    development, validation = train_test_split(
        development_validation,
        test_size=SPLIT_FRACTIONS["validation"] / (1 - SPLIT_FRACTIONS["test"]),
        stratify=labels[development_validation],
        random_state=seed,
    )
    manifest = {
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
        "split_counts": {
            "development": len(development),
            "validation": len(validation),
            "test": len(test),
        },
        "development_indices": [int(index) for index in development],
        "validation_indices": [int(index) for index in validation],
        "test_indices": [int(index) for index in test],
    }
    save_json(path, manifest)
    return manifest


def extract_feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Tạo range tham chiếu cho cảnh báo input đơn biến."""
    ranges: dict[str, dict[str, float]] = {}
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = df[column].dropna()
        ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p0_5": float(values.quantile(0.005)),
            "p99_5": float(values.quantile(0.995)),
        }
    return ranges


def load_data(
    path: str | Path = DEFAULT_RAW_DATA_PATH,
    seed: int = 42,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    return_metadata: bool = False,
) -> tuple[Any, ...]:
    """Trả về feature/label của Development, Validation và Test."""
    raw_df = load_raw_dataset(path)
    clean_df = canonicalize_raw_dataframe(raw_df)
    if TARGET_COLUMN not in clean_df.columns:
        raise KeyError(f"Dataset thiếu cột target: {TARGET_COLUMN}")
    labels = clean_df[TARGET_COLUMN].astype(int)
    if not labels.isin({0, 1}).all():
        raise ValueError("machine_failure chỉ được chứa 0 hoặc 1.")

    failure_modes = [column for column in FAILURE_MODE_COLUMNS if column in clean_df.columns]
    modes = clean_df[failure_modes].copy() if failure_modes else pd.DataFrame(index=clean_df.index)
    features = build_canonical_features(clean_df, expected_features=MODEL_FEATURE_CONTRACT)
    manifest = create_or_load_split_manifest(
        raw_df,
        seed=seed,
        manifest_path=manifest_path,
        dataset_path=path,
    )

    development_indices = manifest["development_indices"]
    validation_indices = manifest["validation_indices"]
    test_indices = manifest["test_indices"]

    def select(indices: list[int]) -> tuple[pd.DataFrame, pd.Series]:
        return (
            features.iloc[indices].reset_index(drop=True),
            labels.iloc[indices].reset_index(drop=True),
        )

    X_development, y_development = select(development_indices)
    X_validation, y_validation = select(validation_indices)
    X_test, y_test = select(test_indices)
    if not return_metadata:
        return X_development, X_validation, X_test, y_development, y_validation, y_test
    return (
        X_development,
        X_validation,
        X_test,
        y_development,
        y_validation,
        y_test,
        modes.iloc[test_indices].reset_index(drop=True),
    )
