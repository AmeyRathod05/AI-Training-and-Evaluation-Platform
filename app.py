from pathlib import Path
from collections.abc import Mapping
from numbers import Integral

import pandas as pd
import streamlit as st

from src.analytics.dashboard import build_analytics, get_category_options
from src.ml.preprocessing import assess_readiness, prepare_training_data
from src.ml.train import train_baseline
from src.validation.validators import (
    ERROR_OPTIONS,
    PREFERENCE_OPTIONS,
    save_annotation as validate_and_save_annotation,
    validate_annotation_collection,
    validate_dataset,
    validate_export,
)

st.set_page_config(
    page_title="AI Training & Evaluation Platform",
    page_icon="📊",
    layout="wide",
)

st.title("AI Training & Evaluation Platform")
st.caption("Human feedback. Measurable quality. Reproducible evaluation.")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "sample_dataset.csv"

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "workflow_message" not in st.session_state:
    st.session_state.workflow_message = None

# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except (OSError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f"Error reading dataset: {exc}")
    st.stop()

dataset_validation = validate_dataset(df)
if dataset_validation.errors:
    for error in dataset_validation.errors:
        st.error(error)
    st.stop()

for warning in dataset_validation.warnings:
    st.warning(warning)

record_ids = df["record_id"].tolist()
annotation_validation = validate_annotation_collection(
    st.session_state.annotations,
    record_ids,
)


def rating_default(value):
    if isinstance(value, Integral) and not isinstance(value, bool) and 1 <= value <= 5:
        return int(value)
    return 3


def annotation_count_for_dataset():
    return len(annotation_validation.annotated_record_ids)

# Initialize the selected record.
if (
    "selected_record_id" not in st.session_state
    or st.session_state.selected_record_id not in record_ids
):
    st.session_state.selected_record_id = record_ids[0]

# --------------------------------------------------
# Save annotation callback
# --------------------------------------------------

def save_annotation():
    record_id = st.session_state.selected_record_id
    key = str(record_id)

    annotation = {
        "record_id": record_id,
        "preference": st.session_state.get(
            f"preference_{key}", "A"
        ),
        "rating_a": st.session_state.get(
            f"rating_a_{key}", 3
        ),
        "rating_b": st.session_state.get(
            f"rating_b_{key}", 3
        ),
        "error_label": st.session_state.get(
            f"error_{key}", "No obvious issue"
        ),
        "justification": st.session_state.get(
            f"justification_{key}", ""
        ),
    }

    validation = validate_and_save_annotation(
        st.session_state.annotations,
        annotation,
        record_ids,
    )
    if not validation.is_valid:
        st.session_state.workflow_message = (
            "error",
            "Annotation was not saved:\n- " + "\n- ".join(validation.errors),
        )
        return

    st.session_state.pop("ml_experiment_result", None)
    st.session_state.workflow_message = (
        "success",
        f"Annotation saved for record {record_id}.",
    )

    # Find the next unannotated record, wrapping around
    # to the beginning when necessary.
    current_index = record_ids.index(record_id)

    ordered_ids = (
        record_ids[current_index + 1:]
        + record_ids[:current_index + 1]
    )

    next_record = next(
        (
            rid
            for rid in ordered_ids
            if str(rid) not in st.session_state.annotations
        ),
        None,
    )

    if next_record is not None:
        st.session_state.selected_record_id = next_record

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Dataset Explorer",
        "Annotation Studio",
        "Annotation Quality",
        "Analytics Dashboard",
        "ML Experiment",
    ]
)

with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Unique Categories", df["category"].nunique())
    col3.metric(
        "Annotated",
        annotation_count_for_dataset(),
        delta=(
            f"{annotation_count_for_dataset() / len(df):.0%} complete"
        ),
    )

    st.dataframe(df, use_container_width=True)

with tab2:
    annotations = st.session_state.annotations
    completed = annotation_count_for_dataset()
    total = len(df)

    st.subheader("Compare AI Responses")
    st.write(
        "Review both responses, use the reference answer where useful, "
        "and record your independent judgment."
    )

    st.progress(min(completed / total, 1.0))
    st.caption(f"{completed} of {total} records annotated")

    # Display messages from the previous save attempt.
    message = st.session_state.workflow_message

    if message:
        message_type, message_text = message

        if message_type == "success":
            st.success(message_text)
        elif message_type == "error":
            st.error(message_text)
        else:
            st.warning(message_text)

        st.session_state.workflow_message = None

    # Record selector
    selected_id = st.selectbox(
        "Select Record ID",
        options=record_ids,
        format_func=lambda value: f"Record {value}",
        key="selected_record_id",
    )

    record = df.loc[df["record_id"] == selected_id].iloc[0]
    key = str(selected_id)
    existing = annotations.get(key, {})
    if not isinstance(existing, Mapping):
        existing = {}

    st.markdown("### Prompt")
    st.info(str(record["prompt"]))

    left, right = st.columns(2)

    with left:
        st.markdown("### Response A")
        st.info(str(record["response_a"]))

    with right:
        st.markdown("### Response B")
        st.info(str(record["response_b"]))

    with st.expander("Reference Answer"):
        st.write(str(record["reference_answer"]))

    st.divider()
    st.markdown("### Annotation Form")

    default_preference = existing.get("preference", "A")

    preference_index = (
        PREFERENCE_OPTIONS.index(default_preference)
        if default_preference in PREFERENCE_OPTIONS
        else 0
    )

    st.radio(
        "Which response do you prefer?",
        options=PREFERENCE_OPTIONS,
        index=preference_index,
        horizontal=True,
        key=f"preference_{key}",
    )

    score_a, score_b = st.columns(2)

    with score_a:
        st.slider(
            "Rate Response A",
            min_value=1,
            max_value=5,
            value=rating_default(existing.get("rating_a", 3)),
            key=f"rating_a_{key}",
        )

    with score_b:
        st.slider(
            "Rate Response B",
            min_value=1,
            max_value=5,
            value=rating_default(existing.get("rating_b", 3)),
            key=f"rating_b_{key}",
        )

    existing_error = existing.get(
        "error_label", "No obvious issue"
    )

    error_index = (
        ERROR_OPTIONS.index(existing_error)
        if existing_error in ERROR_OPTIONS
        else 0
    )

    st.selectbox(
        "Main issue observed",
        options=ERROR_OPTIONS,
        index=error_index,
        key=f"error_{key}",
    )

    st.text_area(
        "Explain your decision",
        value=(
            existing.get("justification", "")
            if isinstance(existing.get("justification", ""), str)
            else ""
        ),
        placeholder="Explain which response is better and why...",
        key=f"justification_{key}",
    )

    st.button(
        "Save annotation & continue",
        type="primary",
        on_click=save_annotation,
    )

    # --------------------------------------------------
    # Saved annotations and export
    # --------------------------------------------------

    st.divider()
    st.subheader("Saved Annotations")

    if annotations:
        csv_export = validate_export(annotations, record_ids, "csv")
        jsonl_export = validate_export(annotations, record_ids, "jsonl")

        st.dataframe(
            pd.DataFrame(list(annotations.values())),
            use_container_width=True,
        )

        if (
            csv_export.is_valid
            and jsonl_export.is_valid
            and csv_export.content is not None
            and jsonl_export.content is not None
        ):
            st.download_button(
                "Export annotations as CSV",
                data=csv_export.content.encode("utf-8"),
                file_name="annotations.csv",
                mime="text/csv",
            )

            st.download_button(
                "Export annotations as JSONL",
                data=jsonl_export.content.encode("utf-8"),
                file_name="annotations.jsonl",
                mime="application/jsonl",
            )
        else:
            st.warning(
                "Exports are disabled because one or more saved annotations "
                "need correction. Review the Annotation Quality tab."
            )
    else:
        st.info(
            "No annotations saved yet. Evaluate your first record above."
        )

    if completed == total:
        st.success(
            "All records have been annotated! "
            "You can review your results or export your dataset."
        )

with tab3:
    st.subheader("Annotation Quality")
    annotated_count = len(annotation_validation.annotated_record_ids)
    pending_count = len(annotation_validation.pending_record_ids)
    valid_count = len(annotation_validation.valid_annotation_keys)
    invalid_count = len(annotation_validation.invalid_annotations)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total dataset records", len(df))
    metric2.metric("Annotated records", annotated_count)
    metric3.metric("Pending records", pending_count)
    metric4.metric("Completion", f"{annotated_count / len(df):.0%}")

    metric5, metric6 = st.columns(2)
    metric5.metric("Valid annotations", valid_count)
    metric6.metric("Invalid annotations", invalid_count)

    if annotation_validation.invalid_annotations:
        st.markdown("#### Validation errors")
        issue_rows = []
        for storage_key, reasons in annotation_validation.invalid_annotations.items():
            saved = st.session_state.annotations.get(storage_key)
            record_id = (
                saved.get("record_id", storage_key)
                if isinstance(saved, Mapping)
                else storage_key
            )
            issue_rows.append(
                {
                    "Record ID": record_id,
                    "Reasons": "; ".join(reasons),
                }
            )
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)
    elif valid_count == 0:
        st.info("No annotations saved yet.")
    else:
        st.success("All saved annotations pass validation.")

    if dataset_validation.warnings:
        st.markdown("#### Dataset warnings")
        for warning in dataset_validation.warnings:
            st.warning(warning)

    if annotation_validation.warnings:
        st.markdown("#### Annotation warnings")
        for warning in annotation_validation.warnings:
            st.warning(warning)

    if annotation_validation.pending_record_ids:
        st.markdown("#### Pending records")
        st.write(", ".join(annotation_validation.pending_record_ids))

with tab4:
    st.subheader("Analytics Dashboard")
    st.caption(
        "Response-quality charts use valid annotations only. "
        "Pending records are not counted as invalid."
    )

    category_options = get_category_options(df)
    selected_category = st.selectbox(
        "Filter by dataset category",
        options=["All categories", *category_options],
        key="analytics_category_filter",
    )
    analytics = build_analytics(
        df,
        st.session_state.annotations,
        category_filter=selected_category,
    )

    st.caption(
        f"Valid annotations included in charts: {analytics.valid_annotations} · "
        f"Invalid annotations excluded: {analytics.invalid_annotations}"
    )

    kpi_columns = st.columns(5)
    kpi_columns[0].metric("Total dataset records", analytics.total_records)
    kpi_columns[1].metric("Annotated records", analytics.annotated_records)
    kpi_columns[2].metric("Pending records", analytics.pending_records)
    kpi_columns[3].metric("Valid annotations", analytics.valid_annotations)
    kpi_columns[4].metric(
        "Annotation completion",
        f"{analytics.completion_percentage:.1f}%",
    )

    if analytics.valid_annotations == 0:
        st.info(
            "No valid annotations match this filter yet. Charts show zero counts "
            "or unavailable averages until a valid annotation is saved."
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Preference distribution")
        st.bar_chart(analytics.preference_counts)

    with chart_right:
        st.markdown("#### Average response ratings")
        if analytics.valid_annotations:
            st.bar_chart(analytics.average_ratings)
        else:
            st.info("Average ratings are unavailable until valid annotations exist.")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Issue label frequency")
        st.bar_chart(analytics.issue_counts)

    with chart_right:
        st.markdown("#### Preference by category")
        st.bar_chart(analytics.category_preference)

with tab5:
    st.subheader("Baseline Preference Experiment")
    st.caption(
        "This experiment uses only valid annotations in the current Streamlit "
        "session. Annotations can disappear when the session resets because "
        "persistent storage has not been implemented."
    )
    st.caption(
        "Minimum readiness threshold: 32 valid labelled examples total, "
        "with at least eight for each preference class."
    )

    prepared = prepare_training_data(df, st.session_state.annotations)
    readiness = assess_readiness(prepared)

    st.markdown("#### Data readiness")
    readiness_columns = st.columns(3)
    readiness_columns[0].metric(
        "Valid labelled examples",
        len(prepared.examples),
    )
    readiness_columns[1].metric(
        "Unique preference classes",
        sum(count > 0 for count in prepared.class_counts.values()),
    )
    readiness_columns[2].metric(
        "Invalid annotations excluded",
        len(prepared.invalid_annotations),
    )

    class_counts_frame = pd.DataFrame(
        {
            "Preference": list(prepared.class_counts),
            "Examples": list(prepared.class_counts.values()),
        }
    ).set_index("Preference")
    st.markdown("**Examples per preference class**")
    st.dataframe(class_counts_frame, use_container_width=True)

    readiness_columns = st.columns(3)
    readiness_columns[0].metric(
        "Records without a valid label",
        len(prepared.missing_label_record_ids),
    )
    readiness_columns[1].metric(
        "Duplicate dataset IDs",
        len(prepared.duplicate_record_ids),
    )
    readiness_columns[2].metric(
        "Duplicate annotation IDs",
        len(prepared.duplicate_annotation_ids),
    )

    if prepared.missing_label_record_ids:
        with st.expander("Records without a valid preference label"):
            st.write(", ".join(prepared.missing_label_record_ids))

    if prepared.invalid_annotations:
        with st.expander("Invalid annotations excluded from training"):
            invalid_rows = [
                {
                    "Annotation key": key,
                    "Reasons": "; ".join(reasons),
                }
                for key, reasons in prepared.invalid_annotations.items()
            ]
            st.dataframe(pd.DataFrame(invalid_rows), use_container_width=True)

    if prepared.duplicate_record_ids:
        st.warning(
            "Rows with duplicate dataset record IDs are excluded. Resolve "
            "these IDs before training."
        )
    if readiness.is_ready:
        st.success(
            "Ready for a fixed-seed stratified 75%/25% train/test split. "
            "At least 32 valid examples total (eight per preference class) "
            "are required. "
            "This is a minimum technical threshold; performance estimates "
            "will still be uncertain with a small dataset."
        )
    else:
        st.warning(
            "Reliable evaluation is not available yet. Collect more valid "
            "annotations rather than merging or inventing preference labels."
        )
        for reason in readiness.reasons:
            st.write(f"- {reason}")

    if st.button(
        "Train baseline model",
        type="primary",
        disabled=not readiness.is_ready,
        key="train_baseline_model",
    ):
        st.session_state.ml_experiment_result = train_baseline(prepared)

    experiment = st.session_state.get("ml_experiment_result")
    if experiment is not None:
        if not experiment.trained:
            st.error(experiment.message)
            for limitation in experiment.limitations:
                st.write(f"- {limitation}")
        else:
            st.success(experiment.message)
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Accuracy",
                f"{experiment.metrics['accuracy']:.3f}",
            )
            metric_columns[1].metric(
                "Macro-F1",
                f"{experiment.metrics['macro_f1']:.3f}",
            )
            metric_columns[2].metric(
                "Test-set size",
                experiment.metrics["test_size"],
            )
            st.markdown("#### Per-class test metrics")
            st.dataframe(
                experiment.per_class_metrics,
                use_container_width=True,
            )
            st.markdown("#### Confusion matrix")
            st.caption("Rows are actual labels; columns are model predictions.")
            st.dataframe(
                experiment.confusion_matrix,
                use_container_width=True,
            )
            st.markdown("#### Test-set class distribution")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Preference": list(experiment.test_class_counts),
                        "Test examples": list(
                            experiment.test_class_counts.values()
                        ),
                    }
                ).set_index("Preference"),
                use_container_width=True,
            )
            st.info(
                "Accuracy is the fraction of held-out examples classified "
                "correctly. Macro-F1 gives each preference class equal weight. "
                "The per-class table and confusion matrix show where errors "
                "occur. This small baseline is experimental, not evidence of "
                "production readiness."
            )
