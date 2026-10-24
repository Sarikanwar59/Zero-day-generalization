from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

BASE = Path("data/raw/unsw_nb15")
RUN_ID = "20261023T211056Z"
MODEL_DIR = Path("models/reproducible_runs") / RUN_ID
OUT_DIR = Path("reports/threshold_diagnostic") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = BASE / "UNSW_NB15_testing-set.csv"
TRAIN_FILE = BASE / "UNSW_NB15_training-set.csv"

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TEST_PER_CLASS = 5000
SEED = 42

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
train_df["attack_cat"] = train_df["attack_cat"].fillna("Unknown").astype(str)

test_df_all = pd.read_csv(TEST_FILE, low_memory=False)
test_df_all["attack_cat"] = test_df_all["attack_cat"].fillna("Unknown").astype(str)
test_df_all["label"] = pd.to_numeric(test_df_all["label"]).astype(int)

results = []

for category in ["Generic", "Exploits"]:
    model_path = MODEL_DIR / f"{category.lower()}_seed_42.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    normal_df = test_df_all[test_df_all["attack_cat"] == "Normal"]
    attack_df = test_df_all[test_df_all["attack_cat"] == category]

    if len(normal_df) < TEST_PER_CLASS or len(attack_df) < TEST_PER_CLASS:
        raise ValueError(
            f"Not enough test rows for {category}: "
            f"normal={len(normal_df)}, attack={len(attack_df)}"
        )

    normal_sample = normal_df.sample(n=TEST_PER_CLASS, random_state=SEED)
    attack_sample = attack_df.sample(n=TEST_PER_CLASS, random_state=SEED)

    evaluation_df = pd.concat(
        [normal_sample, attack_sample], ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    excluded = {"id", "attack_cat", "label"}
    features = [
        col for col in train_df.columns
        if col not in excluded and col in evaluation_df.columns
    ]

    X = evaluation_df[features].copy()
    y = evaluation_df["label"].astype(int).to_numpy()

    model = joblib.load(model_path)
    scores = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)

    for threshold in THRESHOLDS:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        results.append({
            "category": category,
            "threshold": threshold,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "attack_recall": recall_score(y, pred, zero_division=0),
            "f1_score": f1_score(y, pred, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "missed_attacks": int(fn),
            "detected_attacks": int(tp),
            "roc_auc": roc_auc,
            "pr_auc_average_precision": pr_auc,
            "evaluation_use": "diagnostic_only_not_for_threshold_selection",
        })

results_df = pd.DataFrame(results)
csv_path = OUT_DIR / "threshold_results.csv"
results_df.to_csv(csv_path, index=False)

for category in ["Generic", "Exploits"]:
    print(f"\n=== {category}: threshold diagnostic ===")
    subset = results_df[results_df["category"] == category]
    print(subset[[
        "threshold",
        "attack_recall",
        "false_positive_rate",
        "precision",
        "f1_score",
        "missed_attacks",
        "false_positives",
        "roc_auc",
        "pr_auc_average_precision",
    ]].round(4).to_string(index=False))

print(f"\nSaved: {csv_path}")
print("IMPORTANT: These held-out test results are diagnostic only.")
print("Do not select a final threshold from this table.")
