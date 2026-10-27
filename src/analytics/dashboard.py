"""Analytics calculations based on the existing dataset and annotation schema."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.validation.validators import (
    ERROR_OPTIONS,
    PREFERENCE_OPTIONS,
    validate_annotation_collection,
)

ALL_CATEGORIES = "All categories"
MISSING_CATEGORY = "(Missing category)"


@dataclass
class AnalyticsResult:
    total_records: int
    annotated_records: int
    pending_records: int
    valid_annotations: int
    invalid_annotations: int
    completion_percentage: float
    preference_counts: pd.DataFrame
    average_ratings: pd.DataFrame
    issue_counts: pd.DataFrame
    category_preference: pd.DataFrame


def get_category_options(dataframe: pd.DataFrame) -> list[str]:
    """Return normalized, sorted categories, including missing category values."""
    if "category" not in dataframe.columns:
        return []
    return sorted({_category_label(value) for value in dataframe["category"]})


def _record_key(value: Any) -> str | None:
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        return None
    if isinstance(value, bool):
        return None
    return str(value)


def _category_label(value: Any) -> str:
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        return MISSING_CATEGORY
    label = str(value).strip()
    return label if label else MISSING_CATEGORY


def build_analytics(
    dataframe: pd.DataFrame,
    annotations: Any,
    category_filter: str = ALL_CATEGORIES,
) -> AnalyticsResult:
    """Calculate dashboard metrics and chart data from valid annotations only."""
    record_column = "record_id"
    category_column = "category"
    has_record_ids = record_column in dataframe.columns
    has_categories = category_column in dataframe.columns

    record_categories: dict[str, str] = {}
    if has_record_ids:
        for _, row in dataframe.iterrows():
            key = _record_key(row[record_column])
            if key is not None:
                record_categories[key] = (
                    _category_label(row[category_column])
                    if has_categories
                    else MISSING_CATEGORY
                )

    category_by_record = pd.DataFrame(
        {
            "record_key": [
                _record_key(value)
                for value in dataframe[record_column]
            ] if has_record_ids else [],
            "category": [
                _category_label(value)
                for value in dataframe[category_column]
            ] if has_categories else [
                MISSING_CATEGORY for _ in range(len(dataframe))
            ],
        }
    )
    if category_filter != ALL_CATEGORIES:
        category_by_record = category_by_record.loc[
            category_by_record["category"] == category_filter
        ]

    scoped_record_keys = {
        key for key in category_by_record["record_key"].tolist() if key is not None
    }
    total_records = len(category_by_record)

    if has_record_ids:
        collection = validate_annotation_collection(
            annotations,
            dataframe[record_column].tolist(),
        )
    else:
        collection = validate_annotation_collection(annotations, [])

    annotation_items = annotations.items() if isinstance(annotations, Mapping) else ()
    annotated_keys: set[str] = set()
    for storage_key, annotation in annotation_items:
        storage_record_key = str(storage_key)
        annotation_record_key = (
            _record_key(annotation.get(record_column))
            if isinstance(annotation, Mapping)
            else None
        )
        if storage_record_key in record_categories:
            annotated_keys.add(storage_record_key)
        elif annotation_record_key in record_categories:
            annotated_keys.add(annotation_record_key)

    annotated_records = len(annotated_keys & scoped_record_keys)
    pending_records = max(total_records - annotated_records, 0)

    invalid_keys = set(collection.invalid_annotations)
    invalid_in_scope: set[str] = set()
    for storage_key in invalid_keys:
        annotation = annotations.get(storage_key) if isinstance(annotations, Mapping) else None
        annotation_record_key = (
            _record_key(annotation.get(record_column))
            if isinstance(annotation, Mapping)
            else None
        )
        category_key = (
            annotation_record_key
            if annotation_record_key in record_categories
            else storage_key
        )
        if category_filter == ALL_CATEGORIES or category_key in scoped_record_keys:
            invalid_in_scope.add(storage_key)

    valid_rows: list[dict[str, Any]] = []
    if isinstance(annotations, Mapping):
        for key in collection.valid_annotation_keys:
            annotation = annotations[key]
            record_key = _record_key(annotation[record_column])
            if record_key in scoped_record_keys:
                valid_rows.append(
                    {
                        **annotation,
                        "category": record_categories.get(
                            record_key, MISSING_CATEGORY
                        ),
                    }
                )

    valid_annotations = len(valid_rows)
    completion_percentage = (
        annotated_records / total_records * 100 if total_records else 0.0
    )

    preference_series = (
        pd.Series([row["preference"] for row in valid_rows], dtype="string")
        .value_counts()
        .reindex(PREFERENCE_OPTIONS, fill_value=0)
    )
    preference_counts = preference_series.rename("Count").rename_axis(
        "Preference"
    ).to_frame()

    if valid_rows:
        ratings = pd.DataFrame(
            [
                {
                    "Response A": row["rating_a"],
                    "Response B": row["rating_b"],
                }
                for row in valid_rows
            ]
        ).mean()
    else:
        ratings = pd.Series(
            {"Response A": float("nan"), "Response B": float("nan")}
        )
    average_ratings = ratings.rename("Average rating").rename_axis(
        "Response"
    ).to_frame()

    issue_series = (
        pd.Series([row["error_label"] for row in valid_rows], dtype="string")
        .value_counts()
        .reindex(ERROR_OPTIONS, fill_value=0)
    )
    issue_counts = issue_series.rename("Count").rename_axis("Issue label").to_frame()

    scoped_categories = sorted(
        set(category_by_record["category"].tolist())
    )
    if valid_rows:
        category_preferences = pd.crosstab(
            pd.Series(
                [row["category"] for row in valid_rows],
                name="Category",
                dtype="string",
            ),
            pd.Series(
                [row["preference"] for row in valid_rows],
                name="Preference",
                dtype="string",
            ),
        )
    else:
        category_preferences = pd.DataFrame()
    category_preference = category_preferences.reindex(
        index=scoped_categories,
        columns=PREFERENCE_OPTIONS,
        fill_value=0,
    )

    return AnalyticsResult(
        total_records=total_records,
        annotated_records=annotated_records,
        pending_records=pending_records,
        valid_annotations=valid_annotations,
        invalid_annotations=len(invalid_in_scope),
        completion_percentage=completion_percentage,
        preference_counts=preference_counts,
        average_ratings=average_ratings,
        issue_counts=issue_counts,
        category_preference=category_preference,
    )
