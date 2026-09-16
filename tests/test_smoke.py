"""Smoke test cho pipeline ML và API snapshot."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import SensorPayload, app
from src.contracts import FAILURE_MODE_COLUMNS, MODEL_FEATURE_CONTRACT
from src.data import compute_dataset_sha256, load_data
from src.features import (
    add_engineered_features,
    build_canonical_features,
    canonicalize_raw_dataframe,
)
from src.inference import RiskInferenceService
from src.models import compute_classification_metrics

client = TestClient(app)


def sample_payload(record_id: str | None = None) -> dict[str, object]:
    return {
        "record_id": record_id,
        "product_quality_type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }


def test_health_reports_loaded_artifact() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_ready"] is True


def test_live_and_ready_endpoints() -> None:
    live_response = client.get("/live")
    ready_response = client.get("/ready")
    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 200
    assert ready_response.json()["model_ready"] is True


def test_score_returns_small_canonical_response() -> None:
    response = client.post("/score", json=sample_payload("row_1"))
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"record_id", "failure_risk", "decision", "threshold", "warnings", "model"}
    assert 0.0 <= data["failure_risk"] <= 1.0
    assert data["decision"] in {"NO_ALERT", "REVIEW_REQUIRED"}


def test_rank_orders_snapshots_by_risk() -> None:
    low = sample_payload("low")
    high = sample_payload("high")
    high["tool_wear_min"] = 250.0
    response = client.post("/rank", json={"snapshots": [low, high], "top_k": 2})
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["rank"] for item in items] == [1, 2]
    assert items[0]["failure_risk"] >= items[1]["failure_risk"]


def test_api_rejects_target_and_failure_mode_columns() -> None:
    invalid = sample_payload()
    invalid["machine_failure"] = 1
    response = client.post("/score", json=invalid)
    assert response.status_code == 422


def test_sensor_payload_normalizes_quality_type() -> None:
    payload = SensorPayload(**{**sample_payload(), "product_quality_type": " l "})
    assert payload.product_quality_type == "L"


def test_sensor_payload_rejects_unknown_quality_type() -> None:
    with pytest.raises(ValidationError):
        SensorPayload(**{**sample_payload(), "product_quality_type": "X"})


def test_engineered_features_formulas() -> None:
    raw = pd.DataFrame(
        {
            "quality_type": ["M"],
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [1500.0],
            "torque_nm": [40.0],
            "tool_wear_min": [20.0],
        }
    )
    result = add_engineered_features(raw)
    assert result.loc[0, "temperature_delta_k"] == 10.0
    expected_power = 40.0 * (1500.0 * 2.0 * math.pi / 60.0)
    assert result.loc[0, "mechanical_power_w"] == pytest.approx(expected_power)
    assert result.loc[0, "wear_load_interaction"] == 800.0


def test_engineered_features_are_recomputed_from_raw_sensors() -> None:
    data = pd.DataFrame({**sample_payload(), "temperature_delta_k": [999.0]})
    result = build_canonical_features(data)
    assert result.loc[0, "temperature_delta_k"] == 10.0


def test_conflicting_quality_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="mâu thuẫn"):
        canonicalize_raw_dataframe(pd.DataFrame({"Type": ["L"], "product_quality_type": ["H"]}))


def test_failure_modes_are_not_model_features() -> None:
    _, _, X_test, _, _, _, modes_test = load_data(return_metadata=True)
    assert set(modes_test.columns) <= set(FAILURE_MODE_COLUMNS)
    assert not any(column in X_test.columns for column in FAILURE_MODE_COLUMNS)
    assert list(X_test.columns) == list(MODEL_FEATURE_CONTRACT)


def test_dataset_hash_is_stable_across_newline_styles(tmp_path: Path) -> None:
    content = "column_a,column_b\n1,2\n"
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(content.encode("utf-8"))
    crlf_path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    assert compute_dataset_sha256(lf_path) == compute_dataset_sha256(crlf_path)


def test_split_manifest_matches_dataset_hash_and_has_no_overlap() -> None:
    import json

    manifest = json.loads(Path("reports/split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] == compute_dataset_sha256("data/raw/ai4i2020.csv")

    development = set(manifest["development_indices"])
    validation = set(manifest["validation_indices"])
    test = set(manifest["test_indices"])
    assert not development & validation
    assert not development & test
    assert not validation & test


def test_metrics_have_expected_shape() -> None:
    metrics = compute_classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["pr_auc"] > 0.9
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_service_uses_single_artifact_directory() -> None:
    service = RiskInferenceService.get_instance()
    assert service.is_ready
    assert Path(service.artifacts_dir) == Path("artifacts")
