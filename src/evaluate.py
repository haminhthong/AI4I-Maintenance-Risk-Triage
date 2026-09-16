"""Đánh giá cuối trên Test hold-out và phân tích failure mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .contracts import FAILURE_MODE_COLUMNS, FAILURE_MODE_DESCRIPTIONS
from .data import load_data
from .models import (
    compute_calibration_curve_and_ece,
    compute_classification_metrics,
)
from .policy import failure_capture_at_k, queue_precision_at_k
from .utils import LOGGER, save_json, setup_logging

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")


def evaluate_failure_mode_slices(
    modes_df: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Đo recall từng failure mode; các cờ này chỉ dùng sau khi score."""
    report: dict[str, Any] = {}
    for column in FAILURE_MODE_COLUMNS:
        if column not in modes_df.columns:
            continue
        mode_mask = modes_df[column].to_numpy(dtype=int) == 1
        total = int(mode_mask.sum())
        detected = int(predictions[mode_mask].sum()) if total else 0
        recall = detected / total if total else 0.0
        report[column] = {
            "description": FAILURE_MODE_DESCRIPTIONS.get(column, column),
            "total_test_failures": total,
            "detected": detected,
            "missed": total - detected,
            "recall": recall,
            "recall_percent": f"{recall * 100:.2f}%",
        }
    return report


def analyze_twf_slice(
    X_test: pd.DataFrame,
    modes_test: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Tóm tắt TWF được phát hiện và bị bỏ sót."""
    if "failure_twf" not in modes_test.columns:
        return {"status": "failure_twf unavailable"}
    twf = modes_test["failure_twf"].to_numpy(dtype=int) == 1
    report: dict[str, Any] = {
        "total": int(twf.sum()),
        "detected": int((twf & (predictions == 1)).sum()),
        "missed": int((twf & (predictions == 0)).sum()),
    }
    for name, mask in (
        ("detected", twf & (predictions == 1)),
        ("missed", twf & (predictions == 0)),
    ):
        group = X_test.loc[mask]
        report[name] = {
            "count": len(group),
            "tool_wear_mean": float(group["tool_wear_min"].mean()) if len(group) else None,
            "torque_mean": float(group["torque_nm"].mean()) if len(group) else None,
        }
    return report


def evaluate_model_on_test() -> dict[str, Any]:
    """Đánh giá artifact hiện tại trên Test, không tối ưu ngưỡng ở Test."""
    X_development, X_validation, X_test, y_development, y_validation, y_test, modes_test = (
        load_data(return_metadata=True)
    )
    del X_development, X_validation, y_development, y_validation
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    threshold = json.loads((ARTIFACTS_DIR / "threshold.json").read_text(encoding="utf-8"))
    review_threshold = float(threshold["review_threshold"])
    y_true = y_test.to_numpy(dtype=int)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= review_threshold).astype(int)

    ece, calibration_curve = compute_calibration_curve_and_ece(y_true, probabilities)
    primary_metrics = compute_classification_metrics(y_true, probabilities, review_threshold)
    fixed_metrics = compute_classification_metrics(y_true, probabilities, 0.5)
    failure_modes = evaluate_failure_mode_slices(modes_test, predictions)
    test_performance = {
        "pr_auc": primary_metrics["pr_auc"],
        "roc_auc": primary_metrics["roc_auc"],
        "precision": primary_metrics["precision"],
        "recall": primary_metrics["recall"],
        "f1": primary_metrics["f1"],
        "brier": primary_metrics["brier"],
        "ece": ece,
        "review_coverage": primary_metrics["alert_rate"],
        # Chưa có nhật ký quyết định của kỹ thuật viên để đo metric này.
        "priority_override_rate": None,
        "failure_capture_at_1pct": failure_capture_at_k(y_true, probabilities, 0.01),
        "failure_capture_at_2pct": failure_capture_at_k(y_true, probabilities, 0.02),
        "failure_capture_at_3pct": failure_capture_at_k(y_true, probabilities, 0.03),
        "queue_precision_at_1pct": queue_precision_at_k(y_true, probabilities, 0.01),
        "queue_precision_at_2pct": queue_precision_at_k(y_true, probabilities, 0.02),
        "queue_precision_at_3pct": queue_precision_at_k(y_true, probabilities, 0.03),
        "confusion_matrix": primary_metrics["confusion_matrix"],
    }
    report = {
        "dataset": "AI4I 2020",
        "model": metadata.get("model", "unknown"),
        "model_version": metadata.get("model_version", 1),
        "evaluation_protocol": "stratified_holdout_test_not_used_for_selection",
        "test_samples": len(X_test),
        "test_failures": int(y_true.sum()),
        "review_threshold": review_threshold,
        "test_performance": test_performance,
        "threshold_comparison": {
            "validation_f1_threshold": primary_metrics,
            "fixed_0_50": fixed_metrics,
        },
        "failure_mode_analysis": failure_modes,
        "twf_error_analysis": analyze_twf_slice(X_test, modes_test, predictions),
        "calibration_curve": calibration_curve,
    }
    save_json(REPORTS_DIR / "final_test_metrics.json", report)
    LOGGER.info(
        "Test PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f ECE=%.4f",
        test_performance["pr_auc"],
        test_performance["roc_auc"],
        test_performance["brier"],
        test_performance["ece"],
    )
    return report


if __name__ == "__main__":
    setup_logging()
    evaluate_model_on_test()
