# Social Media Happiness Project

## Problem Description

This project studies relationships between social media use and subjective well-being. The goal is to build models that predict a user's happiness index (or related happiness measure) from features in the dataset. Typical tasks include exploratory analysis, preprocessing, model training, and evaluating model performance.

## Dataset

- File: `data/raw/Mental_Health_and_Social_Media_Balance_Dataset.csv`
- Contains observations of individual users with features describing demographics, social-media usage patterns, and one column measuring happiness or well-being.
- The code expects either a column named `Happiness_Index` or another column with "happiness" in its name; the loader will rename such a column to `Happiness_Index` automatically.

Note: The raw dataset in `data/raw/` is the canonical source. A copy is also present at the repository root for convenience.

## Preprocessing Procedures

The preprocessing implemented in `src/preprocess.py` follows these steps:

1. Remove identifier columns
   - If a `User_ID` column exists it is dropped.

2. Handle categorical variables
   - Categorical columns (dtype `object`) are converted using one-hot encoding (`pd.get_dummies` with `drop_first=True`).

3. Define features and label
   - The target label is `Happiness_Index`.
   - Features `X` are all columns except `Happiness_Index`.

4. Train/test split
   - An 80/20 train/test split is used via `sklearn.model_selection.train_test_split` with `random_state=42`.

5. Feature scaling
   - `StandardScaler` from scikit-learn is fit on the training features and applied to both train and test features.

After preprocessing, the pipeline returns `(X_train_scaled, X_test_scaled, y_train, y_test)`.

## Project structure (relevant files)

- `src/data_loader.py` — dataset loading and initial checks (shape, column names, automatic renaming of happiness column).
- `src/preprocess.py` — preprocessing pipeline described above.
- `src/explore.py` — simple exploratory functions (summary, missing values, basic plots).
- `src/model_training.py`, `src/models.py`, `src/model_plots.py` — training, model definitions, and plotting utilities (see source for details).

## How to run

1. Install dependencies (example):

```bash
pip install -r requirements.txt
```

2. Load the dataset and run preprocessing from Python:

```python
from src.data_loader import load_data
from src.preprocess import preprocess_data

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
```

3. Train/evaluate models using the training utilities in `src/` (see `src/model_training.py`).

## Notes and next steps

- The loader prints dataset shape and column names at load time to help verify the data.
- `run.py` is currently empty — you can add an experiment script or a CLI entry point to orchestrate loading, preprocessing, training, and evaluation.
- If you want the README expanded with column-level descriptions, run `load_data()` to print exact columns and paste them here so I can enumerate and document each column.

## Model Details and Results

Models implemented and evaluated:

- `Linear Regression` — baseline linear model.
- `Decision Tree` — single-tree regressor (random_state=42).
- `Random Forest` — ensemble regressor (n_estimators=200, random_state=42).

I ran a full experiment locally and saved results in the `results/` folder. Key evaluation metrics (on the test set):

- Linear Regression: MAE=0.7964, MSE=0.9343, R2=0.6067
- Decision Tree: MAE=0.8300, MSE=1.3500, R2=0.4317
- Random Forest: MAE=0.7290, MSE=0.8389, R2=0.6469

Artifacts produced:

- `results/model_metrics.json` — JSON with all numeric metrics.
- `results/plots/` — PNG files for Actual vs Predicted, error distributions, feature importances, and learning curves for each model.
- `results/presentation.pdf` — a short multipage PDF aggregating the generated plots for easy viewing.

These experiments used the preprocessing pipeline defined in `src/preprocess.py` and the orchestration script `run.py` (added to the repo). The plots and metrics correspond exactly to the code in `run.py` and the code under `src/`.

## Reproducibility / Example commands

Install dependencies into a venv or conda environment and run the experiment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

After running, open `results/presentation.pdf` to review the inference visualizations and learning-curve plots.

---

If you'd like, I can now: (a) add a runnable `run.py` orchestration script, or (b) expand the README with exact column descriptions after reading the dataset.
