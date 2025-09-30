# Model Card — Census Income Classifier (Logistic Regression)

## Model Details
- Algorithm: Logistic Regression (liblinear, max_iter=2000)
- Features: One-hot encoded categoricals + original numerics
- Training: `census_submission/train_model.py`
- Serving: `census_submission/main.py` (FastAPI)

## Intended Use
Educational assignment; not for production or high-stakes decisions.

## Data
- Source: `census_submission/data/census.csv` (add dataset)
- Target: `salary` (<=50K vs >50K)
- Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Split: 80/20

## Metrics (Overall)
After running `python train_model.py`, the script writes `model/metrics.json`.
- Precision: `<fill from JSON>`
- Recall: `<fill from JSON>`
- F1: `<fill from JSON>`

## Slice Metrics
Written to `slice_output.txt` (by education). Paste highlights here.

## Reproduce
1. `cd census_submission`
2. Put dataset at `data/census.csv`
3. `python train_model.py`
4. `uvicorn main:app --reload`
5. `python local_api.py`
6. `pytest -q`
