from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

BASE = Path("data/raw/unsw_nb15")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_FILE = BASE / "UNSW_NB15_training-set.csv"
TEST_FILE = BASE / "UNSW_NB15_testing-set.csv"

SAMPLE_SIZE = 20000
RANDOM_STATE = 42

# Keep memory usage manageable by selecting random rows from each file.
rng = np.random.default_rng(RANDOM_STATE)

train_total = 175341
test_total = 82332

train_rows = set(
    rng.choice(np.arange(1, train_total + 1), size=SAMPLE_SIZE, replace=False)
)
test_rows = set(
    rng.choice(np.arange(1, test_total + 1), size=SAMPLE_SIZE, replace=False)
)

train_df = pd.read_csv(
    TRAIN_FILE,
    skiprows=lambda i: i > 0 and i not in train_rows,
    low_memory=False,
)
test_df = pd.read_csv(
    TEST_FILE,
    skiprows=lambda i: i > 0 and i not in test_rows,
    low_memory=False,
)

if "label" not in train_df or "label" not in test_df:
    raise ValueError("Expected 'label' column was not found.")

# Exclude identifiers and attack-category descriptions from model inputs.
# The binary label is the target.
excluded = {"id", "attack_cat", "label"}
feature_cols = [
    c for c in train_df.columns
    if c not in excluded and c in test_df.columns
]

X_train = train_df[feature_cols].copy()
y_train = pd.to_numeric(train_df["label"]).astype(int)
X_test = test_df[feature_cols].copy()
y_test = pd.to_numeric(test_df["label"]).astype(int)

# Identify categorical and numeric fields from the training sample.
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=150,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )),
])

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Training label counts:")
print(y_train.value_counts().to_dict())
print("Testing label counts:")
print(y_test.value_counts().to_dict())

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "experiment": "unsw_nb15_sampled_binary_baseline",
    "training_file": TRAIN_FILE.name,
    "testing_file": TEST_FILE.name,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "features_used": feature_cols,
    "excluded_columns": sorted(excluded),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}

joblib.dump(model, MODEL_DIR / "unsw_baseline_model.joblib")

with open(REPORT_DIR / "unsw_baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== UNSW-NB15 SAMPLE BASELINE ===")
print(json.dumps(metrics, indent=2))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, pred, zero_division=0))
print("\nSaved model: models/unsw_baseline_model.joblib")
print("Saved metrics: reports/unsw_baseline_metrics.json")
print("Note: sampled binary classification is not an unseen-attack experiment.")
