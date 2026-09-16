"""Pipeline tiền xử lý, mô hình ứng viên và metric cho AI4I."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contracts import CATEGORICAL_FEATURES, MODEL_FEATURE_CONTRACT, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Tạo tiền xử lý cho dữ liệu số và `quality_type`."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )


def build_pipeline(
    classifier: Any,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    scale_numeric: bool = False,
) -> Pipeline:
    """Ghép tiền xử lý và classifier thành một pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def get_candidate_models(
    seed: int = 42,
    feature_names: Sequence[str] = MODEL_FEATURE_CONTRACT,
) -> dict[str, Any]:
    """Tạo LR, RF và HistGB cho một bộ feature cụ thể."""
    numeric_features = tuple(name for name in feature_names if name not in CATEGORICAL_FEATURES)
    categorical_features = tuple(name for name in feature_names if name in CATEGORICAL_FEATURES)

    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            numeric_features,
            categorical_features,
            scale_numeric=True,
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            numeric_features,
            categorical_features,
            scale_numeric=False,
        ),
        "hist_gradient_boosting": build_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
            numeric_features,
            categorical_features,
            scale_numeric=False,
        ),
    }


def calibrate_model(model: Any, method: str = "sigmoid", cv: int = 3) -> Any:
    """Hiệu chỉnh xác suất sau khi đã chọn base model."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Phương pháp calibration phải là 'sigmoid' hoặc 'isotonic'.")
    return CalibratedClassifierCV(model, method=method, cv=cv)


def compute_calibration_curve_and_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Tính ECE và các điểm calibration theo bin xác suất."""
    if len(y_true) == 0:
        return 0.0, []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    points: list[dict[str, float]] = []
    for index in range(n_bins):
        upper = (
            y_prob <= bin_edges[index + 1] if index == n_bins - 1 else y_prob < bin_edges[index + 1]
        )
        mask = (y_prob >= bin_edges[index]) & upper
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(np.mean(y_prob[mask]))
        fraction_positive = float(np.mean(y_true[mask]))
        ece += count / len(y_true) * abs(fraction_positive - mean_probability)
        points.append(
            {
                "bin_lower": float(bin_edges[index]),
                "bin_upper": float(bin_edges[index + 1]),
                "count": count,
                "mean_predicted": mean_probability,
                "fraction_of_positives": fraction_positive,
            }
        )
    return float(ece), points


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Tính metric phân loại tại một threshold đã chọn trên validation."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Nhãn và xác suất phải cùng kích thước và không được rỗng.")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "alert_rate": float(y_pred.mean()),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
