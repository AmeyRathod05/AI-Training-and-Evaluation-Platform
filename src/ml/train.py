"""Train and evaluate a small, reproducible TF-IDF preference baseline."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.ml.preprocessing import (
    PreparedTrainingData,
    assess_readiness,
)
from src.validation.validators import PREFERENCE_OPTIONS

RANDOM_SEED = 42
TEST_SIZE = 0.25


@dataclass
class ExperimentResult:
    trained: bool
    message: str
    model: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_class_counts: dict[str, int] = field(default_factory=dict)
    train_record_ids: list[Any] = field(default_factory=list)
    test_record_ids: list[Any] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def train_baseline(
    prepared: PreparedTrainingData,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> ExperimentResult:
    """Train a pipeline using a held-out split; never fit TF-IDF on test text."""
    readiness = assess_readiness(prepared, test_size=test_size)
    if not readiness.is_ready:
        return ExperimentResult(
            trained=False,
            message="Training is not available yet.",
            limitations=readiness.reasons,
        )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        return ExperimentResult(
            trained=False,
            message=(
                "scikit-learn is required for the ML experiment. "
                "Install the project's declared dependencies and restart the app."
            ),
            limitations=[str(exc)],
        )

    examples = prepared.examples
    all_labels = list(PREFERENCE_OPTIONS)
    try:
        train_indices, test_indices = train_test_split(
            range(len(examples)),
            test_size=test_size,
            random_state=random_seed,
            stratify=examples["preference"],
        )
    except ValueError as exc:
        return ExperimentResult(
            trained=False,
            message="A stratified train/test split could not be created.",
            limitations=[
                f"Add valid examples to each preference class and retry. "
                f"Split details: {exc}"
            ],
        )

    train = examples.iloc[list(train_indices)]
    test = examples.iloc[list(test_indices)]
    train_labels = set(train["preference"])
    test_labels = set(test["preference"])
    missing_train_labels = [
        label for label in all_labels if label not in train_labels
    ]
    missing_test_labels = [
        label for label in all_labels if label not in test_labels
    ]
    if missing_train_labels or missing_test_labels:
        limitations = []
        if missing_train_labels:
            limitations.append(
                "Training split is missing preference labels: "
                + ", ".join(missing_train_labels)
                + "."
            )
        if missing_test_labels:
            limitations.append(
                "Test split is missing preference labels: "
                + ", ".join(missing_test_labels)
                + ". Collect more examples per class for complete evaluation."
            )
        return ExperimentResult(
            trained=False,
            message="Evaluation is unavailable because the split is incomplete.",
            test_class_counts={
                label: int(test["preference"].value_counts().get(label, 0))
                for label in all_labels
            },
            train_record_ids=train["record_id"].tolist(),
            test_record_ids=test["record_id"].tolist(),
            limitations=limitations,
        )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    try:
        model.fit(train["text"], train["preference"])
        predictions = model.predict(test["text"])
    except Exception as exc:
        return ExperimentResult(
            trained=False,
            message="Model training or evaluation failed.",
            limitations=[
                f"{type(exc).__name__}: {exc}"
            ],
            train_record_ids=train["record_id"].tolist(),
            test_record_ids=test["record_id"].tolist(),
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        test["preference"],
        predictions,
        labels=all_labels,
        zero_division=0,
    )
    per_class_metrics = pd.DataFrame(
        {
            "Preference": all_labels,
            "Precision": precision,
            "Recall": recall,
            "F1 score": f1,
            "Test examples": support,
        }
    ).set_index("Preference")
    confusion = confusion_matrix(
        test["preference"],
        predictions,
        labels=all_labels,
    )
    confusion_frame = pd.DataFrame(
        confusion,
        index=pd.Index(all_labels, name="Actual"),
        columns=pd.Index(all_labels, name="Predicted"),
    )
    metrics = {
        "accuracy": float(accuracy_score(test["preference"], predictions)),
        "macro_f1": float(
            f1_score(
                test["preference"],
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "test_size": len(test),
    }
    test_counts = {
        label: int(test["preference"].value_counts().get(label, 0))
        for label in all_labels
    }

    return ExperimentResult(
        trained=True,
        message="Baseline training and held-out evaluation completed.",
        model=model,
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        confusion_matrix=confusion_frame,
        test_class_counts=test_counts,
        train_record_ids=train["record_id"].tolist(),
        test_record_ids=test["record_id"].tolist(),
    )
