"""Prepare validated in-session annotations for a baseline preference model."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from numbers import Integral
from typing import Any

import pandas as pd

from src.validation.validators import (
    PREFERENCE_OPTIONS,
    validate_annotation_collection,
)

TEXT_FIELDS = ("prompt", "response_a", "response_b")
TEXT_SEPARATOR = "\n\n"
MIN_EXAMPLES_PER_CLASS = 8


@dataclass
class PreparedTrainingData:
    examples: pd.DataFrame
    class_counts: dict[str, int]
    missing_label_record_ids: list[str] = field(default_factory=list)
    invalid_annotations: dict[str, list[str]] = field(default_factory=dict)
    duplicate_record_ids: list[str] = field(default_factory=list)
    duplicate_annotation_ids: list[str] = field(default_factory=list)
    invalid_records_excluded: int = 0


@dataclass
class ReadinessResult:
    is_ready: bool
    reasons: list[str]
    minimum_examples_per_class: int
    expected_test_size: int


def _record_key(value: Any) -> str | None:
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    return None


def prepare_training_data(
    dataframe: pd.DataFrame,
    annotations: Any,
) -> PreparedTrainingData:
    """Join valid annotations to their unique dataset rows without mutating input."""
    required_columns = {"record_id", *TEXT_FIELDS}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing ML input columns: "
            + ", ".join(sorted(missing_columns))
        )

    id_counts = Counter(
        key
        for value in dataframe["record_id"]
        if (key := _record_key(value)) is not None
    )
    duplicate_ids = sorted(
        record_id for record_id, count in id_counts.items() if count > 1
    )

    record_rows: dict[str, pd.Series] = {}
    for _, row in dataframe.iterrows():
        record_key = _record_key(row["record_id"])
        if record_key is not None and id_counts[record_key] == 1:
            record_rows[record_key] = row

    record_ids = dataframe["record_id"].tolist()
    collection = validate_annotation_collection(annotations, record_ids)
    valid_keys = set(collection.valid_annotation_keys)

    annotation_keys_by_record: dict[str, list[str]] = {}
    if isinstance(annotations, Mapping):
        for storage_key, annotation in annotations.items():
            if not isinstance(annotation, Mapping):
                continue
            record_key = _record_key(annotation.get("record_id"))
            if record_key is not None:
                annotation_keys_by_record.setdefault(record_key, []).append(
                    str(storage_key)
                )

    missing_labels: list[str] = []
    examples: list[dict[str, Any]] = []
    for record_key, row in record_rows.items():
        matching_keys = annotation_keys_by_record.get(record_key, [])
        if not matching_keys:
            missing_labels.append(record_key)
            continue

        matching_valid_keys = [
            key for key in matching_keys if key in valid_keys
        ]
        if len(matching_valid_keys) != 1:
            if any(
                isinstance(annotations.get(key), Mapping)
                and (
                    not isinstance(annotations[key].get("preference"), str)
                    or not annotations[key].get("preference", "").strip()
                )
                for key in matching_keys
            ):
                missing_labels.append(record_key)
            continue

        annotation = annotations[matching_valid_keys[0]]
        text = TEXT_SEPARATOR.join(
            f"--- {field} ---\n{str(row[field]).strip()}"
            for field in TEXT_FIELDS
        )
        examples.append(
            {
                "record_id": row["record_id"],
                "text": text,
                "preference": annotation["preference"],
            }
        )
    # Duplicate dataset IDs and malformed or mis-keyed annotations are excluded
    # explicitly; never select an arbitrary duplicate row for training.
    for duplicate_id in duplicate_ids:
        if duplicate_id not in missing_labels:
            missing_labels.append(duplicate_id)

    example_frame = pd.DataFrame(
        examples,
        columns=("record_id", "text", "preference"),
    )
    class_counts = (
        example_frame["preference"]
        .value_counts()
        .reindex(PREFERENCE_OPTIONS, fill_value=0)
        .astype(int)
        .to_dict()
        if not example_frame.empty
        else {label: 0 for label in PREFERENCE_OPTIONS}
    )
    duplicate_annotation_ids = sorted(
        key
        for key, reasons in collection.invalid_annotations.items()
        if any("appears more than once" in reason for reason in reasons)
    )

    return PreparedTrainingData(
        examples=example_frame,
        class_counts=class_counts,
        missing_label_record_ids=sorted(set(missing_labels)),
        invalid_annotations=dict(collection.invalid_annotations),
        duplicate_record_ids=duplicate_ids,
        duplicate_annotation_ids=duplicate_annotation_ids,
        invalid_records_excluded=len(collection.invalid_annotations)
        + len(duplicate_ids),
    )


def assess_readiness(
    prepared: PreparedTrainingData,
    test_size: float = 0.25,
    minimum_examples_per_class: int = MIN_EXAMPLES_PER_CLASS,
) -> ReadinessResult:
    """Require all supported labels in train and test with minimum representation."""
    reasons: list[str] = []
    class_count = len(PREFERENCE_OPTIONS)
    examples_count = len(prepared.examples)
    expected_test_size = max(ceil(examples_count * test_size), class_count)
    expected_train_size = examples_count - expected_test_size

    missing_classes = [
        label
        for label in PREFERENCE_OPTIONS
        if prepared.class_counts.get(label, 0) == 0
    ]
    if missing_classes:
        reasons.append(
            "No valid examples for preference labels: "
            + ", ".join(missing_classes)
            + "."
        )

    underrepresented = [
        f"{label} ({prepared.class_counts.get(label, 0)})"
        for label in PREFERENCE_OPTIONS
        if prepared.class_counts.get(label, 0) < minimum_examples_per_class
    ]
    if underrepresented:
        reasons.append(
            f"Collect at least {minimum_examples_per_class} valid examples "
            "for each preference class; currently underrepresented: "
            + ", ".join(underrepresented)
            + "."
        )
    if expected_test_size >= examples_count or expected_train_size < class_count:
        reasons.append(
            "There are too few valid labelled examples for a train/test split "
            "that can represent every preference class."
        )
    if len(prepared.duplicate_record_ids) > 0:
        reasons.append(
            "Duplicate dataset record IDs were excluded; resolve them before training."
        )
    if not examples_count:
        reasons.append("No valid labelled examples are available.")

    return ReadinessResult(
        is_ready=not reasons,
        reasons=reasons,
        minimum_examples_per_class=minimum_examples_per_class,
        expected_test_size=expected_test_size,
    )
