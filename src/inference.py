"""Suy luận một snapshot hoặc xếp hạng một batch snapshot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .contracts import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_CONTRACT,
    TARGET_COLUMN,
)
from .features import build_canonical_features, canonicalize_raw_dataframe
from .input_validation import check_input_ranges
from .policy import decision_from_risk, rank_rows

LOGGER = logging.getLogger("ai4i.inference")
ARTIFACTS_DIR = Path("artifacts")


class RiskInferenceService:
    """Nạp một artifact duy nhất và chấm điểm risk hỏng hóc hiện tại."""

    _instance: RiskInferenceService | None = None

    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.threshold: dict[str, Any] = {}
        self.reference_ranges: dict[str, dict[str, float]] = {}
        self.artifact_checks: dict[str, Any] = {}
        self.is_loaded = False
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> RiskInferenceService:
        """Trả về service dùng chung trong tiến trình API và Streamlit."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        """Artifact phải tải được và có threshold trước khi score."""
        return self.is_loaded and bool(self.artifact_checks.get("ready"))

    def load_artifacts(self) -> None:
        """Nạp model, metadata, threshold và range tham chiếu từ `artifacts/`."""
        required = {
            "model": self.artifacts_dir / "model.joblib",
            "metadata": self.artifacts_dir / "metadata.json",
            "threshold": self.artifacts_dir / "threshold.json",
        }
        self.artifact_checks = {
            "artifacts_dir": str(self.artifacts_dir),
            "model_file": required["model"].is_file(),
            "metadata_file": required["metadata"].is_file(),
            "threshold_file": required["threshold"].is_file(),
            "reference_ranges_file": (self.artifacts_dir / "reference_ranges.json").is_file(),
            "ready": False,
        }
        try:
            if not all(
                self.artifact_checks[key]
                for key in ("model_file", "metadata_file", "threshold_file")
            ):
                return
            self.model = joblib.load(required["model"])
            self.metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
            self.threshold = json.loads(required["threshold"].read_text(encoding="utf-8"))
            ranges_path = self.artifacts_dir / "reference_ranges.json"
            if ranges_path.exists():
                self.reference_ranges = json.loads(ranges_path.read_text(encoding="utf-8"))
            features = self.metadata.get("features", [])
            threshold_value = self.threshold.get("review_threshold")
            self.artifact_checks["feature_contract_exact"] = features == list(
                MODEL_FEATURE_CONTRACT
            )
            self.artifact_checks["threshold_valid"] = (
                isinstance(threshold_value, int | float) and 0 <= threshold_value <= 1
            )
            self.artifact_checks["ready"] = bool(
                self.model is not None
                and self.artifact_checks["feature_contract_exact"]
                and self.artifact_checks["threshold_valid"]
            )
            self.is_loaded = self.model is not None
        except (OSError, TypeError, ValueError) as exc:
            self.is_loaded = False
            self.artifact_checks["error"] = str(exc)
            LOGGER.exception("Không thể nạp artifact: %s", exc)

    def predict(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Chuyển raw sensor snapshot thành risk, decision và warnings."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Artifact mô hình chưa sẵn sàng.")

        canonical_payload = canonicalize_raw_dataframe(pd.DataFrame([raw_payload]))
        forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN} | set(FAILURE_MODE_COLUMNS)
        leaked_columns = sorted(forbidden & set(canonical_payload.columns))
        if leaked_columns:
            raise ValueError(f"Payload chứa cột không được phép: {leaked_columns}")

        features = build_canonical_features(
            canonical_payload, expected_features=MODEL_FEATURE_CONTRACT
        )
        failure_risk = float(self.model.predict_proba(features)[0, 1])
        review_threshold = float(self.threshold["review_threshold"])
        warnings = check_input_ranges(features, self.reference_ranges)
        return {
            "record_id": raw_payload.get("record_id"),
            "failure_risk": round(failure_risk, 6),
            "decision": decision_from_risk(failure_risk, review_threshold),
            "threshold": review_threshold,
            "warnings": warnings,
            "model": self.metadata.get("model", "unknown"),
        }

    def rank(
        self, raw_payloads: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Chấm điểm cả batch bằng một lần gọi `predict_proba`."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Artifact mô hình chưa sẵn sàng.")
        if not raw_payloads:
            return []

        rows = []
        for index, payload in enumerate(raw_payloads):
            row = dict(payload)
            row.setdefault("record_id", f"row_{index}")
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        record_ids = raw_df["record_id"].tolist()
        model_input = raw_df.drop(columns=["record_id"], errors="ignore")
        canonical_df = canonicalize_raw_dataframe(model_input)

        forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN} | set(FAILURE_MODE_COLUMNS)
        leaked_columns = sorted(forbidden & set(canonical_df.columns))
        if leaked_columns:
            raise ValueError(f"Payload chứa cột không được phép: {leaked_columns}")

        features = build_canonical_features(canonical_df, expected_features=MODEL_FEATURE_CONTRACT)
        probabilities = self.model.predict_proba(features)[:, 1]
        review_threshold = float(self.threshold["review_threshold"])
        model_name = self.metadata.get("model", "unknown")

        scored = []
        for index, probability in enumerate(probabilities):
            risk_value = float(probability)
            scored.append(
                {
                    "record_id": record_ids[index],
                    "failure_risk": round(risk_value, 6),
                    "decision": decision_from_risk(risk_value, review_threshold),
                    "threshold": review_threshold,
                    "warnings": check_input_ranges(features.iloc[[index]], self.reference_ranges),
                    "model": model_name,
                }
            )
        return rank_rows(scored, top_k=top_k)
