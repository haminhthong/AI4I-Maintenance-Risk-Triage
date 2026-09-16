"""Dashboard Streamlit tối giản cho scoring snapshot và xếp hạng batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.contracts import (
    FAILURE_MODE_COLUMNS,
    FAILURE_MODE_DESCRIPTIONS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from src.features import canonicalize_raw_dataframe
from src.inference import RiskInferenceService

st.set_page_config(
    page_title="AI4I Maintenance Risk Triage",
    page_icon="⚙️",
    layout="wide",
)

SCOPE_TEXT = (
    "Phạm vi: triage rủi ro theo operating snapshot hiện tại. "
    "Hệ thống không ước tính RUL, time-to-failure hoặc xác suất hỏng trong tương lai."
)


@st.cache_data
def load_json_report(filename: str) -> dict[str, Any]:
    """Đọc một báo cáo JSON đã được sinh bởi pipeline đánh giá."""
    path = Path("reports") / filename
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_overview(service: RiskInferenceService, report: dict[str, Any]) -> None:
    """Hiển thị metric locked test và chỉ số phục vụ hàng đợi."""
    st.title("AI4I Maintenance Risk Triage")
    st.caption(
        "Chấm điểm rủi ro từ snapshot cảm biến, sau đó xếp hạng hàng đợi kiểm tra "
        "theo năng lực kỹ thuật viên."
    )

    threshold = float(service.threshold.get("review_threshold", 0.0))
    metadata = service.metadata
    overview = st.columns(4)
    overview[0].metric("Trạng thái", "READY" if service.is_ready else "UNAVAILABLE")
    overview[1].metric("Model", str(metadata.get("model", "unknown")))
    overview[2].metric("Threshold", f"{threshold:.4f}")
    overview[3].metric("Test snapshots", f"{report.get('test_samples', 0):,}")

    if not report:
        st.warning("Chưa có reports/final_test_metrics.json. Hãy chạy pipeline đánh giá.")
        st.info(SCOPE_TEXT)
        return

    performance = report.get("test_performance", {})
    st.subheader("Locked test")
    metric_specs = (
        ("PR-AUC", "pr_auc", ".4f"),
        ("ROC-AUC", "roc_auc", ".4f"),
        ("Brier", "brier", ".4f"),
        ("ECE", "ece", ".4f"),
        ("Precision", "precision", ".2%"),
        ("Recall", "recall", ".2%"),
    )
    metric_columns = st.columns(len(metric_specs))
    for column, (label, key, format_spec) in zip(metric_columns, metric_specs, strict=True):
        value = performance.get(key)
        column.metric(label, "n/a" if value is None else format(float(value), format_spec))

    st.subheader("Capacity-aware queue")
    queue_rows = []
    for percentage in (1, 2, 3):
        capture = performance.get(f"failure_capture_at_{percentage}pct")
        precision = performance.get(f"queue_precision_at_{percentage}pct")
        if capture is not None or precision is not None:
            queue_rows.append(
                {
                    "Năng lực kiểm tra": f"Top {percentage}%",
                    "Failure Capture@K": ("n/a" if capture is None else f"{float(capture):.2%}"),
                    "Queue Precision@K": (
                        "n/a" if precision is None else f"{float(precision):.2%}"
                    ),
                }
            )
    if queue_rows:
        st.dataframe(pd.DataFrame(queue_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Báo cáo chưa có metric Top-K.")
    st.info(SCOPE_TEXT)


def render_single_snapshot(service: RiskInferenceService) -> None:
    """Nhận một snapshot cảm biến và trả về risk, decision cùng warning."""
    st.subheader("Single Snapshot Prediction")
    st.caption(
        "Nhập 6 biến vận hành thô; 3 engineered features sẽ được tính trong pipeline dùng chung."
    )
    with st.form("single_snapshot_form"):
        left, right = st.columns(2)
        with left:
            record_id = st.text_input("Record ID", value="SNAPSHOT_01")
            quality_type = st.selectbox("Product quality type", ["L", "M", "H"], index=1)
            air_temperature = st.number_input("Air temperature [K]", 280.0, 330.0, 300.0, 0.1)
            process_temperature = st.number_input(
                "Process temperature [K]", 280.0, 340.0, 310.0, 0.1
            )
        with right:
            rotational_speed = st.number_input(
                "Rotational speed [rpm]", 500.0, 4000.0, 1500.0, 10.0
            )
            torque = st.number_input("Torque [Nm]", 0.0, 120.0, 40.0, 0.5)
            tool_wear = st.number_input("Tool wear [min]", 0.0, 350.0, 50.0, 1.0)
        submitted = st.form_submit_button("Chấm điểm rủi ro", type="primary")

    if not submitted:
        return

    payload = {
        "record_id": record_id.strip() or "SNAPSHOT_01",
        "product_quality_type": quality_type,
        "air_temperature_k": air_temperature,
        "process_temperature_k": process_temperature,
        "rotational_speed_rpm": rotational_speed,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
    }
    try:
        result = service.predict(payload)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        st.error(f"Không thể chấm điểm snapshot: {exc}")
        return

    result_columns = st.columns(3)
    result_columns[0].metric("Failure risk", f"{result['failure_risk']:.2%}")
    result_columns[1].metric("Decision", result["decision"])
    result_columns[2].metric("Threshold", f"{result['threshold']:.4f}")
    if result["warnings"]:
        st.warning("; ".join(result["warnings"]))
    else:
        st.success("Snapshot nằm trong dải tham chiếu đầu vào.")

    with st.expander("Engineered features"):
        st.json(
            {
                "temperature_delta_k": process_temperature - air_temperature,
                "mechanical_power_w": (torque * rotational_speed * 2.0 * 3.141592653589793 / 60.0),
                "wear_load_interaction": tool_wear * torque,
            }
        )


def _prepare_batch_records(input_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Chuẩn hóa batch và loại identifier trước khi gọi service.rank."""
    clean_df = canonicalize_raw_dataframe(input_df)
    forbidden = [TARGET_COLUMN, *FAILURE_MODE_COLUMNS]
    leaked = [column for column in forbidden if column in clean_df.columns]
    if leaked:
        raise ValueError(f"File chứa cột hậu nghiệm không được phép: {leaked}")

    if "record_id" in clean_df.columns:
        record_ids = clean_df["record_id"].astype(str)
    elif "udi" in clean_df.columns:
        record_ids = clean_df["udi"].astype(str)
    elif "product_id" in clean_df.columns:
        record_ids = clean_df["product_id"].astype(str)
    else:
        record_ids = pd.Series([f"ROW_{index:04d}" for index in range(len(clean_df))])

    model_df = clean_df.drop(columns=["record_id", *IDENTIFIER_COLUMNS], errors="ignore")
    records = model_df.to_dict(orient="records")
    for record, record_id in zip(records, record_ids, strict=True):
        record["record_id"] = record_id
    return records


def render_batch_ranking(service: RiskInferenceService) -> None:
    """Xếp hạng toàn bộ batch và hiển thị phần đầu theo năng lực kiểm tra."""
    st.subheader("Batch Risk Ranking")
    st.caption("CSV có thể dùng tên cột AI4I gốc hoặc tên canonical của API.")
    uploaded_file = st.file_uploader("Tải CSV snapshot", type=["csv"])
    top_k = st.number_input("Số dòng hiển thị trong queue", min_value=1, max_value=1000, value=20)
    if uploaded_file is None:
        st.info("Tải một CSV chỉ gồm biến vận hành để bắt đầu.")
        return

    try:
        input_df = pd.read_csv(uploaded_file)
        records = _prepare_batch_records(input_df)
        ranked = service.rank(records, top_k=None)
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Không thể xử lý batch: {exc}")
        return

    if not ranked:
        st.warning("Batch không có snapshot hợp lệ.")
        return
    ranked_df = pd.DataFrame(ranked)
    display_df = ranked_df.head(int(top_k)).copy()
    display_df["warning_count"] = display_df["warnings"].map(len)
    st.dataframe(
        display_df[["rank", "record_id", "failure_risk", "decision", "warning_count"]],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Tải toàn bộ ranked queue",
        data=ranked_df.to_csv(index=False).encode("utf-8"),
        file_name="ai4i_ranked_queue.csv",
        mime="text/csv",
    )


def render_model_details(report: dict[str, Any], ablation: dict[str, Any]) -> None:
    """Hiển thị failure-mode slices, ablation và giới hạn đã biết."""
    st.subheader("Model Performance and Limitations")
    performance = report.get("test_performance", {})
    confusion_matrix = performance.get("confusion_matrix")
    if confusion_matrix:
        st.write("Confusion matrix trên locked test:")
        st.dataframe(
            pd.DataFrame(
                confusion_matrix,
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            ),
            use_container_width=True,
        )

    st.markdown("#### Critical slices")
    slices = report.get("failure_mode_analysis", {})
    rows = []
    for column in FAILURE_MODE_COLUMNS:
        values = slices.get(column)
        if values:
            rows.append(
                {
                    "Failure mode": column.removeprefix("failure_"),
                    "Mô tả": FAILURE_MODE_DESCRIPTIONS.get(column, column),
                    "Test failures": values.get("total_test_failures", 0),
                    "Recall": values.get("recall_percent", "n/a"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    twf = report.get("twf_error_analysis", {})
    if twf:
        st.warning(
            f"TWF: phát hiện {twf.get('detected', 0)}/{twf.get('total', 0)} ca; "
            "đây là slice yếu cần được xem cùng quy trình kiểm tra mòn dụng cụ."
        )

    if ablation:
        st.markdown("#### Feature ablation")
        st.json(ablation)
    st.info(SCOPE_TEXT)


def main() -> None:
    """Khởi chạy dashboard sau khi kiểm tra artifact."""
    service = RiskInferenceService.get_instance()
    if not service.is_ready:
        st.error("Artifact chưa sẵn sàng. Chạy `python -m src.train` trước khi mở dashboard.")
        return

    report = load_json_report("final_test_metrics.json")
    ablation = load_json_report("feature_ablation.json")
    overview, single, batch, details = st.tabs(
        ["Tổng quan", "Single Snapshot", "Batch Ranking", "Hiệu năng & giới hạn"]
    )
    with overview:
        render_overview(service, report)
    with single:
        render_single_snapshot(service)
    with batch:
        render_batch_ranking(service)
    with details:
        render_model_details(report, ablation)


if __name__ == "__main__":
    main()
