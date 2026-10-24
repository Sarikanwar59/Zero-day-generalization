from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
OUT_DIR = Path("reports/model_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = BASE / "UNSW_NB15_training-set.csv"
SEED = 42
SAMPLE_SIZE = 20000
TARGET_FPR = 0.05

df = pd.read_csv(TRAIN_FILE, low_memory=False)
df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)
df["label"] = pd.to_numeric(df["label"]).astype(int)

def threshold_for_fpr(normal_scores, target_fpr):
    """Choose a threshold using calibration NORMAL records only."""
    scores = np.sort(np.asarray(normal_scores))
    allowed_fp = int(np.floor(target_fpr * len(scores)))

    # Strictly above the selected score limits calibration false alarms.
    if allowed_fp == 0:
        return float(np.nextafter(scores[-1], np.inf))

    boundary_index = len(scores) - allowed_fp - 1
    return float(np.nextafter(scores[boundary_index], np.inf))

model_specs = {
    "LogisticRegression": LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    ),
}

all_results = []

for held_out in ["Generic", "Exploits"]:
    print(f"\n=== Preparing validation experiment: {held_out} ===")

    pool = df[df["attack_cat"] != held_out].copy()

    if len(pool) < SAMPLE_SIZE:
        raise ValueError(f"Not enough rows after excluding {held_out}")

    sample = pool.sample(n=SAMPLE_SIZE, random_state=SEED)
    sample = sample.reset_index(drop=True)

    excluded = {"id", "attack_cat", "label"}
    features = [
        col for col in sample.columns if col not in excluded
    ]

    X = sample[features].copy()
    y = sample["label"].astype(int)
    categories = sample["attack_cat"]

    # Split into fit, threshold-calibration, and validation sets.
    X_fit, X_rest, y_fit, y_rest, cat_fit, cat_rest = train_test_split(
        X, y, categories,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    X_cal, X_eval, y_cal, y_eval, cat_cal, cat_eval = train_test_split(
        X_rest, y_rest, cat_rest,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=y_rest,
    )

    # Fit preprocessing using the fitting partition only.
    categorical = X_fit.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric = [c for c in features if c not in categorical]

    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    print(
        f"Rows: fit={len(X_fit)}, calibration={len(X_cal)}, "
        f"validation={len(X_eval)}"
    )
    print("Held-out category in sampled data:", held_out not in set(categories))
    print("Fit label counts:", y_fit.value_counts().to_dict())

    for model_name, classifier in model_specs.items():
        start = time.time()

        model = Pipeline([
            ("preprocess", clone(preprocess)),
            ("classifier", clone(classifier)),
        ])

        model.fit(X_fit, y_fit)

        # Threshold is calibrated on NORMAL calibration records only.
        normal_cal_mask = (cat_cal == "Normal").to_numpy()
        if normal_cal_mask.sum() == 0:
            raise ValueError("No normal records in calibration split")

        cal_scores = model.predict_proba(X_cal)[:, 1]
        threshold = threshold_for_fpr(
            cal_scores[normal_cal_mask], TARGET_FPR
        )

        # Evaluate on a separate validation partition.
        eval_scores = model.predict_proba(X_eval)[:, 1]
        eval_pred = (eval_scores >= threshold).astype(int)
        y_eval_np = y_eval.to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            y_eval_np, eval_pred, labels=[0, 1]
        ).ravel()

        normal_eval_mask = (cat_eval == "Normal").to_numpy()
        normal_eval_count = int(normal_eval_mask.sum())
        normal_eval_fp = int(
            ((eval_pred == 1) & normal_eval_mask).sum()
        )
        eval_fpr = (
            normal_eval_fp / normal_eval_count
            if normal_eval_count else None
        )

        result = {
            "held_out_category": held_out,
            "model": model_name,
            "seed": SEED,
            "target_calibration_fpr": TARGET_FPR,
            "selected_threshold": threshold,
            "fit_rows": int(len(X_fit)),
            "calibration_rows": int(len(X_cal)),
            "validation_rows": int(len(X_eval)),
            "held_out_category_in_fit": False,
            "held_out_category_in_calibration": False,
            "held_out_category_in_validation": False,
            "validation_accuracy": float(
                accuracy_score(y_eval_np, eval_pred)
            ),
            "validation_precision": float(
                precision_score(y_eval_np, eval_pred, zero_division=0)
            ),
            "validation_recall_known_attacks": float(
                recall_score(y_eval_np, eval_pred, zero_division=0)
            ),
            "validation_f1": float(
                f1_score(y_eval_np, eval_pred, zero_division=0)
            ),
            "validation_false_positive_rate": eval_fpr,
            "validation_normal_rows": normal_eval_count,
            "validation_false_positives": normal_eval_fp,
            "validation_roc_auc": float(
                roc_auc_score(y_eval_np, eval_scores)
            ),
            "validation_average_precision": float(
                average_precision_score(y_eval_np, eval_scores)
            ),
            "validation_confusion_matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
            "elapsed_seconds": round(time.time() - start, 2),
            "interpretation": (
                "Validation uses known categories only. "
                "It does not measure held-out-category recall."
            ),
        }

        all_results.append(result)

        print(
            f"{model_name}: threshold={threshold:.6f}, "
            f"known-attack recall={result['validation_recall_known_attacks']:.4f}, "
            f"validation FPR={eval_fpr:.4f}, "
            f"ROC-AUC={result['validation_roc_auc']:.4f}, "
            f"seconds={result['elapsed_seconds']}"
        )

results_df = pd.DataFrame(all_results)
csv_path = OUT_DIR / "validation_model_comparison.csv"
json_path = OUT_DIR / "validation_model_comparison.json"

results_df.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump({
        "experiment": "validation_model_comparison",
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "target_calibration_fpr": TARGET_FPR,
        "scikit_learn_version": sklearn.__version__,
        "dataset_file": str(TRAIN_FILE),
        "results": all_results,
        "limitations": [
            "The held-out attack category is excluded from all partitions.",
            "Threshold calibration uses normal calibration records only.",
            "Validation measures known attack categories, not held-out attacks.",
            "This experiment does not establish real-world zero-day detection.",
        ],
    }, f, indent=2)

print("\n=== VALIDATION MODEL COMPARISON ===")
print(results_df[[
    "held_out_category",
    "model",
    "selected_threshold",
    "validation_recall_known_attacks",
    "validation_false_positive_rate",
    "validation_f1",
    "validation_roc_auc",
    "validation_average_precision",
]].round(4).to_string(index=False))

print(f"\nSaved CSV:  {csv_path}")
print(f"Saved JSON: {json_path}")
print("\nThis experiment does not use the testing-set CSV.")
