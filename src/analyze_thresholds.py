from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

BASE = Path("data/raw/unsw_nb15")
OUT = Path("reports/threshold_analysis")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = BASE / "UNSW_NB15_testing-set.csv"
TEST_FILE = BASE / "UNSW_NB15_training-set.csv"
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

results = []

for category in ["Generic", "Exploits"]:
    print(f"\nAnalyzing held-out category: {category}")

    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    train_df["attack_cat"] = train_df["attack_cat"].fillna("Unknown").astype(str)

    # Reconstruct the same held-out evaluation sample used by the original script.
    normal_parts = []
    attack_parts = []

    for chunk in pd.read_csv(TEST_FILE, chunksize=10000, low_memory=False):
        chunk["attack_cat"] = chunk["attack_cat"].fillna("Unknown").astype(str)
        normal = chunk[chunk["attack_cat"] == "Normal"]
        attack = chunk[chunk["attack_cat"] == category]

        if len(normal):
            normal_parts.append(normal)
        if len(attack):
            attack_parts.append(attack)

    normal_df = pd.concat(normal_parts, ignore_index=True)
    attack_df = pd.concat(attack_parts, ignore_index=True)

    normal_df = normal_df.sample(n=5000, random_state=42)
    attack_df = attack_df.sample(n=5000, random_state=42)

    test_df = pd.concat([normal_df, attack_df], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    excluded = {"id", "attack_cat", "label"}
    features = [c for c in train_df.columns if c not in excluded and c in test_df.columns]
    X_test = test_df[features].copy()
    y_test = pd.to_numeric(test_df["label"]).astype(int).to_numpy()

    model_path = Path("models") / f"unseen_{category.lower()}_model.joblib"
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    print(f"ROC-AUC: {auc:.4f}")

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        row = {
            "category": category,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "attack_recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": auc,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "missed_attacks": int(fn),
            "detected_attacks": int(tp),
        }
        results.append(row)

df = pd.DataFrame(results)
csv_path = OUT / "threshold_results.csv"
md_path = OUT / "threshold_results.md"

df.to_csv(csv_path, index=False)
with open(md_path, "w") as f:
    f.write("# Exploratory Threshold Analysis\n\n")
    f.write("Thresholds are evaluated on the same fixed samples used in the existing experiments. These results are exploratory, not independent validation.\n\n")
    f.write(df.round(4).to_markdown(index=False))

print("\n=== THRESHOLD RESULTS ===")
print(df.round(4).to_string(index=False))
print(f"\nSaved: {csv_path}")
print(f"Saved: {md_path}")
