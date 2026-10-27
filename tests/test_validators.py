import json
from io import StringIO

import pandas as pd
import pytest

from src.validation.validators import (
    ANNOTATION_FIELDS,
    REQUIRED_COLUMNS,
    save_annotation,
    validate_annotation,
    validate_annotation_collection,
    validate_dataset,
    validate_export,
)


@pytest.fixture
def dataset():
    return pd.DataFrame(
        [
            {
                "record_id": 1,
                "prompt": "Explain a reliable software test.",
                "response_a": "A clear response that explains testing.",
                "response_b": "A different response with enough detail.",
                "reference_answer": "Tests verify expected behavior.",
                "category": "software",
            },
            {
                "record_id": 2,
                "prompt": "Describe a useful code review.",
                "response_a": "Reviews find issues and improve clarity.",
                "response_b": "Review code to improve its quality.",
                "reference_answer": "A review evaluates a proposed change.",
                "category": "software",
            },
        ]
    )


@pytest.fixture
def valid_annotation():
    return {
        "record_id": 1,
        "preference": "A",
        "rating_a": 4,
        "rating_b": 3,
        "error_label": "No obvious issue",
        "justification": "Response A is more specific.",
    }


def test_valid_dataset_passes_validation(dataset):
    result = validate_dataset(dataset)

    assert result.is_valid
    assert not result.errors


def test_missing_required_columns_are_detected(dataset):
    result = validate_dataset(dataset.drop(columns=["response_b"]))

    assert not result.is_valid
    assert any("response_b" in error for error in result.errors)


def test_empty_dataset_is_reported(dataset):
    result = validate_dataset(dataset.iloc[0:0])

    assert not result.is_valid
    assert any("empty" in error.lower() for error in result.errors)


def test_duplicate_record_ids_are_detected(dataset):
    duplicated = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)

    result = validate_dataset(duplicated)

    assert not result.is_valid
    assert any("duplicate record_id" in error.lower() for error in result.errors)


def test_whitespace_only_required_values_are_detected(dataset):
    dataset.loc[0, "prompt"] = " \t "

    result = validate_dataset(dataset)

    assert not result.is_valid
    assert any("prompt" in error and "blank" in error for error in result.errors)


def test_valid_annotation_passes_validation(valid_annotation):
    result = validate_annotation(valid_annotation, [1, 2])

    assert result.is_valid
    assert not result.errors


def test_empty_justification_is_rejected(valid_annotation):
    valid_annotation["justification"] = " \n "

    result = validate_annotation(valid_annotation, [1, 2])

    assert not result.is_valid
    assert any("justification" in error for error in result.errors)


@pytest.mark.parametrize("preference", ["C", "", None])
def test_invalid_preference_is_rejected(valid_annotation, preference):
    valid_annotation["preference"] = preference

    result = validate_annotation(valid_annotation, [1, 2])

    assert any("preference" in error for error in result.errors)


@pytest.mark.parametrize("error_label", ["Unknown issue", "", None])
def test_invalid_issue_label_is_rejected(valid_annotation, error_label):
    valid_annotation["error_label"] = error_label

    result = validate_annotation(valid_annotation, [1, 2])

    assert any("error_label" in error for error in result.errors)


@pytest.mark.parametrize("field,value", [("rating_a", 0), ("rating_b", 6)])
def test_out_of_range_ratings_are_rejected(valid_annotation, field, value):
    valid_annotation[field] = value

    result = validate_annotation(valid_annotation, [1, 2])

    assert any(field in error for error in result.errors)


def test_unknown_record_id_is_rejected(valid_annotation):
    valid_annotation["record_id"] = 99

    result = validate_annotation(valid_annotation, [1, 2])

    assert any("not in the loaded dataset" in error for error in result.errors)


def test_editing_existing_annotation_updates_in_place(valid_annotation):
    annotations = {"1": dict(valid_annotation)}
    edited = dict(valid_annotation, preference="Tie", rating_a=2)

    result = save_annotation(annotations, edited, [1, 2])

    assert result.is_valid
    assert len(annotations) == 1
    assert annotations["1"]["preference"] == "Tie"
    assert annotations["1"]["rating_a"] == 2


def test_invalid_annotation_does_not_overwrite_existing(valid_annotation):
    annotations = {"1": dict(valid_annotation)}
    original = dict(annotations["1"])
    invalid = dict(valid_annotation, justification="  ")

    result = save_annotation(annotations, invalid, [1, 2])

    assert not result.is_valid
    assert annotations == {"1": original}


@pytest.mark.parametrize("export_format", ["csv", "jsonl"])
def test_valid_exports_parse_successfully(valid_annotation, export_format):
    result = validate_export({"1": valid_annotation}, [1, 2], export_format)

    assert result.is_valid
    assert result.content is not None
    if export_format == "csv":
        parsed = pd.read_csv(StringIO(result.content))
        assert tuple(parsed.columns) == ANNOTATION_FIELDS
        assert len(parsed) == 1
    else:
        parsed = [json.loads(line) for line in result.content.splitlines()]
        assert len(parsed) == 1
        assert tuple(parsed[0]) == ANNOTATION_FIELDS


def test_invalid_annotation_prevents_export(valid_annotation):
    invalid = dict(valid_annotation, preference="Neither")

    result = validate_export({"1": invalid}, [1, 2], "csv")

    assert not result.is_valid
    assert result.content is None
    assert any("preference" in error for error in result.errors)


def test_unannotated_records_are_pending_not_invalid(dataset):
    result = validate_annotation_collection({}, dataset["record_id"].tolist())

    assert not result.invalid_annotations
    assert result.pending_record_ids == ["1", "2"]
    assert result.annotated_record_ids == []


def test_expected_dataset_schema_constant_matches_app_fields():
    assert REQUIRED_COLUMNS == (
        "record_id",
        "prompt",
        "response_a",
        "response_b",
        "reference_answer",
        "category",
    )


def test_dataset_warnings_do_not_discard_short_records(dataset):
    dataset.loc[0, "response_a"] = "Short"

    result = validate_dataset(dataset)

    assert result.is_valid
    assert any("response_a" in warning for warning in result.warnings)
    assert len(dataset) == 2
