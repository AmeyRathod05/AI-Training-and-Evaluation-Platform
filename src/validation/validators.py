"""Validation helpers for datasets, annotations, and annotation exports."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from io import StringIO
import json
from numbers import Integral
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    "record_id",
    "prompt",
    "response_a",
    "response_b",
    "reference_answer",
    "category",
)

PREFERENCE_OPTIONS = ("A", "B", "Tie", "Both poor")

ERROR_OPTIONS = (
    "No obvious issue",
    "Factual error",
    "Irrelevant answer",
    "Instruction-following failure",
    "Unclear or incomplete",
    "Potential safety issue",
)

ANNOTATION_FIELDS = (
    "record_id",
    "preference",
    "rating_a",
    "rating_b",
    "error_label",
    "justification",
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass
class AnnotationCollectionResult(ValidationResult):
    invalid_annotations: dict[str, list[str]] = field(default_factory=dict)
    valid_annotation_keys: list[str] = field(default_factory=list)
    annotated_record_ids: list[str] = field(default_factory=list)
    pending_record_ids: list[str] = field(default_factory=list)


@dataclass
class ExportValidationResult(ValidationResult):
    content: str | None = None


def _is_missing(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def _record_id_key(value: Any) -> str | None:
    if _is_missing(value) or not pd.api.types.is_scalar(value):
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    return None


def validate_dataset(dataframe: pd.DataFrame) -> ValidationResult:
    """Validate required dataset fields without modifying or dropping rows."""
    result = ValidationResult()
    if not isinstance(dataframe, pd.DataFrame):
        result.errors.append("Dataset must be a pandas DataFrame.")
        return result

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        result.errors.append(
            f"Missing required columns: {', '.join(missing_columns)}."
        )
    if dataframe.empty:
        result.errors.append("Dataset is empty.")
        return result

    for column in REQUIRED_COLUMNS:
        if column not in dataframe.columns:
            continue
        for row_index, value in dataframe[column].items():
            record_label = f"row {row_index}"
            if column == "record_id" and not _is_missing(value):
                record_label = f"record {value}"

            if _is_missing(value):
                result.errors.append(
                    f"{record_label}: required field '{column}' is missing."
                )
                continue

            if isinstance(value, str) and not value.strip():
                result.errors.append(
                    f"{record_label}: required field '{column}' is blank."
                )
            elif not pd.api.types.is_scalar(value):
                result.errors.append(
                    f"{record_label}: field '{column}' must contain a scalar value."
                )
            elif column == "record_id" and _record_id_key(value) is None:
                result.errors.append(
                    f"{record_label}: field 'record_id' must be a non-empty "
                    "string or integer."
                )
            elif column != "record_id" and not isinstance(value, str):
                result.warnings.append(
                    f"{record_label}: field '{column}' is {type(value).__name__}, "
                    "not text."
                )

    if "record_id" in dataframe.columns:
        non_missing_ids = dataframe["record_id"].dropna()
        duplicate_mask = dataframe["record_id"].duplicated(keep=False)
        for value in dataframe.loc[duplicate_mask, "record_id"].tolist():
            result.errors.append(f"Duplicate record_id value: {value}.")

        normalized_ids: dict[str, list[Any]] = defaultdict(list)
        for value in non_missing_ids.tolist():
            normalized = _record_id_key(value)
            if normalized is not None:
                normalized_ids[normalized].append(value)
        for normalized, values in normalized_ids.items():
            if len(values) > 1 and not all(value == values[0] for value in values):
                result.errors.append(
                    "Record IDs collide when stored by the app as text: "
                    f"{values}."
                )

    sample_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in ("record_id", "prompt")
    ] + ["prompt"]
    if all(column in dataframe.columns for column in sample_columns):
        sample_rows: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        for _, row in dataframe.iterrows():
            values = tuple(row[column] for column in sample_columns)
            if all(pd.api.types.is_scalar(value) for value in values):
                sample_rows[values].append(row.get("record_id"))
        for ids in sample_rows.values():
            if len(ids) > 1:
                result.warnings.append(
                    f"Duplicate sample content found for record IDs {ids}."
                )

    for column, minimum_length in (
        ("prompt", 10),
        ("response_a", 20),
        ("response_b", 20),
    ):
        if column not in dataframe.columns:
            continue
        for row_index, value in dataframe[column].items():
            if isinstance(value, str) and value.strip() and len(value.strip()) < minimum_length:
                record_id = (
                    dataframe.at[row_index, "record_id"]
                    if "record_id" in dataframe.columns
                    else row_index
                )
                result.warnings.append(
                    f"Record {record_id}: '{column}' is unusually short "
                    f"({len(value.strip())} characters)."
                )

    return result


def validate_annotation(
    annotation: Any,
    valid_record_ids: Iterable[Any] | None = None,
) -> ValidationResult:
    """Validate one annotation and optionally confirm its record exists."""
    result = ValidationResult()
    if not isinstance(annotation, Mapping):
        result.errors.append("Annotation must be a mapping of field names to values.")
        return result

    missing_fields = [field for field in ANNOTATION_FIELDS if field not in annotation]
    extra_fields = [field for field in annotation if field not in ANNOTATION_FIELDS]
    if missing_fields:
        result.errors.append(
            f"Missing annotation fields: {', '.join(missing_fields)}."
        )
    if extra_fields:
        result.errors.append(
            f"Unexpected annotation fields: {', '.join(map(str, extra_fields))}."
        )

    if "record_id" in annotation:
        record_key = _record_id_key(annotation["record_id"])
        if record_key is None:
            result.errors.append(
                "Field 'record_id' must be a non-empty string or integer."
            )
        elif valid_record_ids is not None:
            known_ids = {
                key
                for value in valid_record_ids
                if (key := _record_id_key(value)) is not None
            }
            if record_key not in known_ids:
                result.errors.append(
                    f"Record ID '{annotation['record_id']}' is not in the loaded dataset."
                )

    if "preference" in annotation and (
        not isinstance(annotation["preference"], str)
        or annotation["preference"] not in PREFERENCE_OPTIONS
    ):
        result.errors.append(
            f"Field 'preference' must be one of: {', '.join(PREFERENCE_OPTIONS)}."
        )

    for field_name in ("rating_a", "rating_b"):
        if field_name not in annotation:
            continue
        rating = annotation[field_name]
        if (
            isinstance(rating, bool)
            or not isinstance(rating, Integral)
            or not 1 <= rating <= 5
        ):
            result.errors.append(
                f"Field '{field_name}' must be an integer from 1 through 5."
            )

    if "error_label" in annotation and (
        not isinstance(annotation["error_label"], str)
        or annotation["error_label"] not in ERROR_OPTIONS
    ):
        result.errors.append(
            f"Field 'error_label' must be one of: {', '.join(ERROR_OPTIONS)}."
        )

    if "justification" in annotation:
        justification = annotation["justification"]
        if not isinstance(justification, str) or not justification.strip():
            result.errors.append(
                "Field 'justification' must contain non-whitespace text."
            )

    return result


def save_annotation(
    annotations: MutableMapping[str, Any],
    annotation: Mapping[str, Any],
    valid_record_ids: Iterable[Any],
) -> ValidationResult:
    """Validate then update the session annotation mapping in place."""
    result = validate_annotation(annotation, valid_record_ids)
    if not result.is_valid:
        return result

    record_key = _record_id_key(annotation["record_id"])
    if record_key is None:
        result.errors.append("Field 'record_id' cannot be used as an annotation key.")
        return result

    annotations[record_key] = dict(annotation)
    return result


def validate_annotation_collection(
    annotations: Any,
    valid_record_ids: Iterable[Any],
) -> AnnotationCollectionResult:
    """Validate every saved annotation, retaining per-annotation reasons."""
    result = AnnotationCollectionResult()
    if not isinstance(annotations, Mapping):
        result.errors.append("Saved annotations must be a mapping.")
        return result

    valid_record_ids = tuple(valid_record_ids)
    dataset_keys = {
        key
        for value in valid_record_ids
        if (key := _record_id_key(value)) is not None
    }
    entries: list[tuple[str, Any, list[str]]] = []
    ids_to_entries: dict[str, list[int]] = defaultdict(list)

    for storage_key, annotation in annotations.items():
        key = str(storage_key)
        annotation_result = validate_annotation(annotation, valid_record_ids)
        reasons = list(annotation_result.errors)
        if not isinstance(storage_key, str):
            reasons.append("Annotation storage key must be a string.")
        if not isinstance(annotation, Mapping):
            annotation_id = None
        else:
            annotation_id = _record_id_key(annotation.get("record_id"))
            if annotation_id is not None:
                ids_to_entries[annotation_id].append(len(entries))
            if annotation_id != key:
                reasons.append(
                    f"Storage key '{key}' does not match the annotation record_id."
                )
        entries.append((key, annotation, reasons))

    for record_key, entry_indexes in ids_to_entries.items():
        if len(entry_indexes) > 1:
            for index in entry_indexes:
                entries[index][2].append(
                    f"Record ID '{record_key}' appears more than once."
                )

    for key, annotation, reasons in entries:
        if key not in dataset_keys and not any(
            "not in the loaded dataset" in reason for reason in reasons
        ):
            reasons.append(f"Storage key '{key}' is not in the loaded dataset.")
        if reasons:
            result.invalid_annotations[key] = reasons
            result.errors.extend(
                f"Annotation {key}: {reason}" for reason in reasons
            )
        else:
            result.valid_annotation_keys.append(key)

    stored_dataset_keys = {
        str(storage_key)
        for storage_key in annotations
        if str(storage_key) in dataset_keys
    }
    result.annotated_record_ids = sorted(stored_dataset_keys)
    result.pending_record_ids = sorted(dataset_keys - stored_dataset_keys)
    return result


def validate_export(
    annotations: Any,
    valid_record_ids: Iterable[Any],
    export_format: str,
) -> ExportValidationResult:
    """Validate annotations and generate a parse-checked CSV or JSONL export."""
    collection_result = validate_annotation_collection(
        annotations, valid_record_ids
    )
    result = ExportValidationResult(
        errors=list(collection_result.errors),
        warnings=list(collection_result.warnings),
    )
    if not isinstance(annotations, Mapping):
        return result
    if not annotations:
        result.errors.append("No annotations are available to export.")
        return result
    if export_format not in ("csv", "jsonl"):
        result.errors.append("Export format must be 'csv' or 'jsonl'.")
        return result
    if not collection_result.is_valid:
        return result

    rows = [dict(annotations[key]) for key in collection_result.valid_annotation_keys]
    dataframe = pd.DataFrame(rows, columns=ANNOTATION_FIELDS)
    if tuple(dataframe.columns) != ANNOTATION_FIELDS:
        result.errors.append("Export columns do not match the annotation schema.")
        return result

    if export_format == "csv":
        content = dataframe.to_csv(index=False)
        parsed = pd.read_csv(StringIO(content), dtype={"record_id": "string"})
        if tuple(parsed.columns) != ANNOTATION_FIELDS or len(parsed) != len(rows):
            result.errors.append("Generated CSV failed schema or row-count validation.")
            return result
    else:
        content = dataframe.to_json(
            orient="records",
            lines=True,
            force_ascii=False,
        )
        try:
            parsed_rows = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]
        except (TypeError, ValueError):
            result.errors.append("Generated JSONL could not be parsed.")
            return result
        if len(parsed_rows) != len(rows) or any(
            tuple(row.keys()) != ANNOTATION_FIELDS for row in parsed_rows
        ):
            result.errors.append("Generated JSONL failed schema or row-count validation.")
            return result

    result.content = content
    return result
