from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = Path("data/raw/network_traffic.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "failed_connections",
]
target = "label"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )),
])

print("Training baseline model...")
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features": features,
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}

joblib.dump(model, MODEL_DIR / "baseline_model.joblib")

with open(REPORT_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== BASELINE MODEL RESULTS ===")
print(json.dumps(metrics, indent=2))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, pred, zero_division=0))
print("\nModel saved: models/baseline_model.joblib")
print("Metrics saved: reports/baseline_metrics.json")
