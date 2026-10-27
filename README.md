# AI Training & Evaluation Platform

A Streamlit application for reviewing AI response pairs, collecting human
preference annotations, examining annotation quality, and evaluating a
baseline preference classifier.

## Features

- Explore and validate the sample response dataset.
- Compare two responses and save preferences, quality ratings, issue labels,
  and written justifications.
- Review annotation completeness and validation issues.
- View annotation analytics by category.
- Export valid annotations as CSV or JSONL.
- Train and evaluate a TF-IDF and logistic-regression baseline on valid
  annotations.

The application keeps annotations in the current Streamlit session; they are
not persisted after the session resets. The baseline experiment requires at
least 32 valid labeled examples, including at least eight examples for each
preference class. Its results are intended for experimentation, not as evidence
of production readiness.

## Requirements

- Python 3.11 or later
- Packages listed in `requirements.txt`

## Setup

From the project directory, create and activate a virtual environment in
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the application

```powershell
streamlit run app.py
```

The application loads `data/sample_dataset.csv` by default.

## Run tests

```powershell
python -m pytest -q
```

## Dataset format

The CSV dataset must include `record_id`, `prompt`, `response_a`, `response_b`,
`reference_answer`, and `category`. Record IDs must be unique. The ML experiment
uses the prompt and both responses as input; the reference answer is not used
as a model feature.
