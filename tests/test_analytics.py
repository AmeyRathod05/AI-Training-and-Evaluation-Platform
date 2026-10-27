import math

import pandas as pd
import pytest

from src.analytics.dashboard import (
    ALL_CATEGORIES,
    MISSING_CATEGORY,
    build_analytics,
    get_category_options,
)
from src.validation.validators import PREFERENCE_OPTIONS


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "category": ["writing", "writing", "math"],
        }
    )


def annotation(record_id, preference, rating_a, rating_b, issue):
    return {
        "record_id": record_id,
        "preference": preference,
        "rating_a": rating_a,
        "rating_b": rating_b,
        "error_label": issue,
        "justification": "A short explanation.",
    }


@pytest.fixture
def annotations():
    return {
        "1": annotation(1, "A", 5, 3, "No obvious issue"),
        "2": annotation(2, "B", 3, 5, "Factual error"),
    }


def test_empty_dataset_and_annotations_return_zero_safe_metrics():
    result = build_analytics(
        pd.DataFrame(columns=["record_id", "category"]),
        {},
    )

    assert result.total_records == 0
    assert result.annotated_records == 0
    assert result.pending_records == 0
    assert result.valid_annotations == 0
    assert result.invalid_annotations == 0
    assert result.completion_percentage == 0
    assert result.preference_counts["Count"].sum() == 0
    assert result.issue_counts["Count"].sum() == 0
    assert result.category_preference.empty
    assert result.average_ratings["Average rating"].isna().all()


def test_empty_annotations_classify_dataset_records_as_pending(dataset):
    result = build_analytics(dataset, {})

    assert result.total_records == 3
    assert result.annotated_records == 0
    assert result.pending_records == 3
    assert result.valid_annotations == 0
    assert result.invalid_annotations == 0


def test_category_filter_limits_metrics_and_charts(dataset, annotations):
    result = build_analytics(dataset, annotations, category_filter="writing")

    assert result.total_records == 2
    assert result.annotated_records == 2
    assert result.pending_records == 0
    assert result.valid_annotations == 2
    assert result.preference_counts.loc["A", "Count"] == 1
    assert result.preference_counts.loc["B", "Count"] == 1
    assert list(result.category_preference.index) == ["writing"]
    assert result.category_preference.loc["writing", "A"] == 1


def test_preference_counts_include_all_options(dataset):
    all_annotations = {
        "1": annotation(1, "A", 5, 3, "No obvious issue"),
        "2": annotation(2, "B", 3, 5, "Factual error"),
        "3": annotation(3, "Tie", 4, 4, "Unclear or incomplete"),
    }

    result = build_analytics(dataset, all_annotations)

    assert tuple(result.preference_counts.index) == PREFERENCE_OPTIONS
    assert result.preference_counts["Count"].to_dict() == {
        "A": 1,
        "B": 1,
        "Tie": 1,
        "Both poor": 0,
    }


def test_average_ratings_use_only_valid_annotations(dataset, annotations):
    result = build_analytics(dataset, annotations)

    assert result.average_ratings.loc["Response A", "Average rating"] == 4
    assert result.average_ratings.loc["Response B", "Average rating"] == 4


def test_invalid_annotations_are_excluded_from_calculations(dataset, annotations):
    annotations["3"] = annotation(3, "not an allowed preference", 1, 5, "Factual error")

    result = build_analytics(dataset, annotations)

    assert result.valid_annotations == 2
    assert result.invalid_annotations == 1
    assert result.pending_records == 0
    assert result.preference_counts["Count"].sum() == 2
    assert math.isclose(
        result.average_ratings.loc["Response A", "Average rating"],
        4,
    )
    assert result.issue_counts["Count"].sum() == 2


def test_category_filter_scopes_invalid_annotation_count(dataset, annotations):
    annotations["3"] = annotation(3, "not an allowed preference", 1, 5, "Factual error")

    writing = build_analytics(dataset, annotations, category_filter="writing")
    math = build_analytics(dataset, annotations, category_filter="math")

    assert writing.invalid_annotations == 0
    assert math.invalid_annotations == 1
    assert math.valid_annotations == 0


def test_missing_categories_are_grouped_and_filterable():
    dataset_with_missing_category = pd.DataFrame(
        {"record_id": [1, 2], "category": [None, ""]}
    )
    result = build_analytics(
        dataset_with_missing_category,
        {"1": annotation(1, "A", 4, 2, "No obvious issue")},
        category_filter=MISSING_CATEGORY,
    )

    assert MISSING_CATEGORY in get_category_options(dataset_with_missing_category)
    assert result.total_records == 2
    assert result.valid_annotations == 1
    assert result.category_preference.loc[MISSING_CATEGORY, "A"] == 1


@pytest.mark.parametrize("category_filter", [ALL_CATEGORIES, "absent"])
def test_filter_with_no_matching_category_is_safe(dataset, category_filter):
    result = build_analytics(dataset, {}, category_filter=category_filter)

    if category_filter == ALL_CATEGORIES:
        assert result.total_records == 3
    else:
        assert result.total_records == 0
    assert result.valid_annotations == 0
    assert result.completion_percentage == 0
