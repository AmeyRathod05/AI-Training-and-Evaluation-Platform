import pandas as pd
import sklearn.model_selection

from src.ml.preprocessing import assess_readiness, prepare_training_data
from src.ml.train import train_baseline
from src.validation.validators import PREFERENCE_OPTIONS


def suitable_dataset(examples_per_class=8):
    rows = []
    annotations = {}
    for label_index, label in enumerate(PREFERENCE_OPTIONS):
        for sample_index in range(examples_per_class):
            record_id = label_index * 100 + sample_index
            unique_token = f"onlyinrecord{record_id}word"
            rows.append(
                {
                    "record_id": record_id,
                    "prompt": f"prompt class{label_index} {unique_token}",
                    "response_a": f"response alpha class{label_index}",
                    "response_b": f"response beta class{label_index}",
                    "reference_answer": "Excluded reference text.",
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


def test_training_and_test_records_do_not_overlap():
    dataframe, annotations = suitable_dataset()
    prepared = prepare_training_data(dataframe, annotations)
    result = train_baseline(prepared)

    assert result.trained
    assert set(result.train_record_ids).isdisjoint(result.test_record_ids)
    assert len(result.train_record_ids) + len(result.test_record_ids) == len(
        prepared.examples
    )


def test_tfidf_vocabulary_is_fitted_only_on_training_text():
    dataframe, annotations = suitable_dataset()
    prepared = prepare_training_data(dataframe, annotations)
    result = train_baseline(prepared)

    assert result.trained
    train_text = set(
        prepared.examples.loc[
            prepared.examples["record_id"].isin(result.train_record_ids),
            "text",
        ]
    )
    test_rows = prepared.examples.loc[
        prepared.examples["record_id"].isin(result.test_record_ids)
    ]
    vocabulary = result.model.named_steps["tfidf"].vocabulary_
    train_tokens = set(" ".join(train_text).split())
    test_only_tokens = set(" ".join(test_rows["text"]).split()) - train_tokens

    assert test_only_tokens
    assert test_only_tokens.isdisjoint(vocabulary)


def test_suitable_dataset_produces_requested_evaluation_metrics():
    dataframe, annotations = suitable_dataset()
    result = train_baseline(prepare_training_data(dataframe, annotations))

    assert result.trained
    assert 0 <= result.metrics["accuracy"] <= 1
    assert 0 <= result.metrics["macro_f1"] <= 1
    assert result.metrics["test_size"] == len(result.test_record_ids)
    assert list(result.per_class_metrics.index) == list(PREFERENCE_OPTIONS)
    assert result.confusion_matrix.shape == (4, 4)
    assert sum(result.test_class_counts.values()) == result.metrics["test_size"]
    assert all(count > 0 for count in result.test_class_counts.values())


def test_missing_classes_prevent_training_with_actionable_limitation():
    dataframe, annotations = suitable_dataset()
    unavailable_label = PREFERENCE_OPTIONS[-1]
    annotations = {
        key: value
        for key, value in annotations.items()
        if value["preference"] != unavailable_label
    }
    prepared = prepare_training_data(dataframe, annotations)
    readiness = assess_readiness(prepared)
    result = train_baseline(prepared)

    assert not readiness.is_ready
    assert any(unavailable_label in reason for reason in readiness.reasons)
    assert not result.trained
    assert result.metrics == {}


def test_training_failure_returns_explicit_result():
    dataframe, annotations = suitable_dataset()
    prepared = prepare_training_data(dataframe, annotations)
    prepared.examples.loc[:, "text"] = ""

    result = train_baseline(prepared)

    assert not result.trained
    assert result.message == "Model training or evaluation failed."
    assert any("empty vocabulary" in reason.lower() for reason in result.limitations)


def test_stratified_split_failure_is_reported(monkeypatch):
    dataframe, annotations = suitable_dataset()

    def fail_split(*args, **kwargs):
        raise ValueError(
            f"synthetic split failure ({len(args)} positional, "
            f"{len(kwargs)} keyword arguments)"
        )

    monkeypatch.setattr(
        sklearn.model_selection,
        "train_test_split",
        fail_split,
    )
    result = train_baseline(prepare_training_data(dataframe, annotations))

    assert not result.trained
    assert result.message == "A stratified train/test split could not be created."
    assert any("synthetic split failure" in reason for reason in result.limitations)
