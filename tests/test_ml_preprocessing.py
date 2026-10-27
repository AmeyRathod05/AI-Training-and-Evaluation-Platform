import pandas as pd
import pytest

from src.ml.preprocessing import (
    MIN_EXAMPLES_PER_CLASS,
    assess_readiness,
    prepare_training_data,
)
from src.validation.validators import PREFERENCE_OPTIONS


@pytest.fixture
def training_dataset():
    rows = []
    annotations = {}
    for label_index, label in enumerate(PREFERENCE_OPTIONS):
        for sample_index in range(5):
            record_id = label_index * 10 + sample_index
            rows.append(
                {
                    "record_id": record_id,
                    "prompt": f"Prompt class{label_index} sample{sample_index}",
                    "response_a": (
                        f"ResponseA class{label_index} sample{sample_index}"
                    ),
                    "response_b": (
                        f"ResponseB class{label_index} sample{sample_index}"
                    ),
                    "reference_answer": "Not a model feature.",
                    "category": f"category{label_index}",
                }
            )
            annotations[str(record_id)] = {
                "record_id": record_id,
                "preference": label,
                "rating_a": 4,
                "rating_b": 3,
                "error_label": "No obvious issue",
                "justification": "Synthetic test-only annotation.",
            }
    return pd.DataFrame(rows), annotations


def test_annotations_join_to_matching_dataset_rows(training_dataset):
    dataframe, annotations = training_dataset
    prepared = prepare_training_data(dataframe, annotations)

    assert len(prepared.examples) == 20
    assert set(prepared.examples["record_id"]) == set(dataframe["record_id"])
    for row in prepared.examples.itertuples(index=False):
        dataset_row = dataframe.loc[
            dataframe["record_id"] == row.record_id
        ].iloc[0]
        assert "--- prompt ---" in row.text
        assert "--- response_a ---" in row.text
        assert "--- response_b ---" in row.text
        assert dataset_row["prompt"] in row.text
        assert dataset_row["response_a"] in row.text
        assert dataset_row["response_b"] in row.text
        assert dataset_row["reference_answer"] not in row.text
        assert row.preference == annotations[str(row.record_id)]["preference"]
    assert list(prepared.examples.columns) == [
        "record_id",
        "text",
        "preference",
    ]


def test_preparation_excludes_invalid_annotations(training_dataset):
    dataframe, annotations = training_dataset
    annotations["0"]["preference"] = "not a supported label"

    prepared = prepare_training_data(dataframe, annotations)

    assert len(prepared.examples) == 19
    assert prepared.invalid_annotations["0"]
    assert "0" not in set(prepared.examples["record_id"].astype(str))
    assert prepared.invalid_records_excluded >= 1


def test_empty_and_insufficient_data_are_not_ready(training_dataset):
    dataframe, _ = training_dataset
    empty = prepare_training_data(dataframe, {})
    insufficient_annotations = {
        str(record_id): {
            "record_id": record_id,
            "preference": "A",
            "rating_a": 3,
            "rating_b": 3,
            "error_label": "No obvious issue",
            "justification": "Synthetic test-only annotation.",
        }
        for record_id in dataframe["record_id"].head(3)
    }
    insufficient = prepare_training_data(dataframe, insufficient_annotations)

    assert not assess_readiness(empty).is_ready
    assert not assess_readiness(insufficient).is_ready


def test_missing_preference_labels_are_reported(training_dataset):
    dataframe, annotations = training_dataset
    annotations.pop("1")
    annotations["0"]["preference"] = ""

    prepared = prepare_training_data(dataframe, annotations)

    assert "1" in prepared.missing_label_record_ids
    assert "0" in prepared.missing_label_record_ids
    assert "0" in prepared.invalid_annotations
    assert prepared.class_counts["A"] == 3


def test_class_counts_include_all_preference_classes(training_dataset):
    dataframe, annotations = training_dataset
    prepared = prepare_training_data(dataframe, annotations)

    assert tuple(prepared.class_counts) == PREFERENCE_OPTIONS
    assert all(
        count == 5
        for count in prepared.class_counts.values()
    )
    assert assess_readiness(prepared).minimum_examples_per_class == (
        MIN_EXAMPLES_PER_CLASS
    )


def test_duplicate_dataset_ids_are_reported_and_excluded(training_dataset):
    dataframe, annotations = training_dataset
    duplicate_row = dataframe.iloc[[0]].copy()
    duplicated_data = pd.concat([dataframe, duplicate_row], ignore_index=True)

    prepared = prepare_training_data(duplicated_data, annotations)

    assert prepared.duplicate_record_ids == ["0"]
    assert "0" not in set(prepared.examples["record_id"].astype(str))
    assert not assess_readiness(prepared).is_ready


def test_preparation_does_not_mutate_dataset_or_annotations(training_dataset):
    dataframe, annotations = training_dataset
    original_data = dataframe.copy(deep=True)
    original_annotations = {key: dict(value) for key, value in annotations.items()}

    prepare_training_data(dataframe, annotations)

    pd.testing.assert_frame_equal(dataframe, original_data)
    assert annotations == original_annotations
