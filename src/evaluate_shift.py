from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

MODEL_PATH = Path("models/baseline_model.joblib")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the model trained on the original synthetic dataset
model = joblib.load(MODEL_PATH)

# Generate a shifted synthetic evaluation dataset
rng = np.random.default_rng(123)
n = 2000

normal = pd.DataFrame({
    "duration": rng.exponential(0.8, n),
    "src_bytes": rng.lognormal(7.8, 1.3, n),
    "dst_bytes": rng.lognormal(7.0, 1.2, n),
    "packet_count": rng.poisson(45, n),
    "failed_connections": rng.poisson(1.2, n),
    "label": 0,
})

attack = pd.DataFrame({
    "duration": rng.exponential(0.6, n),
    "src_bytes": rng.lognormal(6.5, 1.3, n),
    "dst_bytes": rng.lognormal(6.3, 1.3, n),
    "packet_count": rng.poisson(42, n),
    "failed_connections": rng.poisson(2.0, n),
    "label": 1,
})

shifted = pd.concat([normal, attack], ignore_index=True)
shifted = shifted.sample(frac=1, random_state=123).reset_index(drop=True)

features = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "failed_connections",
]

X_shift = shifted[features]
y_shift = shifted["label"]

pred = model.predict(X_shift)
prob = model.predict_proba(X_shift)[:, 1]

metrics = {
    "evaluation": "synthetic_distribution_shift",
    "records": int(len(shifted)),
    "accuracy": float(accuracy_score(y_shift, pred)),
    "precision": float(precision_score(y_shift, pred, zero_division=0)),
    "recall": float(recall_score(y_shift, pred, zero_division=0)),
    "f1_score": float(f1_score(y_shift, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_shift, prob)),
    "confusion_matrix": confusion_matrix(y_shift, pred).tolist(),
}

with open(REPORT_DIR / "shift_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== DISTRIBUTION SHIFT RESULTS ===")
print(json.dumps(metrics, indent=2))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_shift, pred, zero_division=0))
print("\nSaved: reports/shift_metrics.json")
print("Note: this is synthetic evaluation, not real zero-day validation.")
