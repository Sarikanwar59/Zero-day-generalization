from pathlib import Path
from datetime import datetime, timezone
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

BASE = Path("data/raw/unsw_nb15")
TRAIN_FILE = BASE / "UNSW_NB15_training-set.csv"
TEST_FILE = BASE / "UNSW_NB15_testing-set.csv"

SEED = 42
TRAIN_SAMPLE = 20000
TEST_PER_CLASS = 5000
TARGET_FPR = 0.05

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path("reports/rf_calibrated") / RUN_ID
MODEL_DIR = Path("models/rf_calibrated") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def threshold_for_fpr(normal_scores, target_fpr):
    scores = np.sort(np.asarray(normal_scores, dtype=float))
    if len(scores) == 0:
        raise ValueError("No normal calibration scores available.")

    allowed_fp = int(np.floor(target_fpr * len(scores)))
    if allowed_fp == 0:
        return float(np.nextafter(scores[-1], np.inf))

    boundary_index = len(scores) - allowed_fp - 1
    return float(np.nextafter(scores[boundary_index], np.inf))


def load_data():
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)

    for df in (train_df, test_df):
        df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)
        df["label"] = pd.to_numeric(df["label"]).astype(int)

    return train_df, test_df


def make_model(X_fit):
    excluded = {"id", "attack_cat", "label"}
    features = [c for c in X_fit.columns if c not in excluded]

    categorical = X_fit[features].select_dtypes(
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

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )),
    ])
    return model, features


train_df, test_df = load_data()
all_results = []

for held_out in ["Generic", "Exploits"]:
    print(f"\n=== Random Forest: held-out category = {held_out} ===")
    start = time.time()

    # Exclude the target attack category from all training-side partitions.
    pool = train_df[train_df["attack_cat"] != held_out].copy()
    if len(pool) < TRAIN_SAMPLE:
        raise ValueError(f"Not enough training rows after excluding {held_out}.")

    sample = pool.sample(n=TRAIN_SAMPLE, random_state=SEED).reset_index(drop=True)
    X = sample.drop(columns=["id", "attack_cat", "label"], errors="ignore")
    y = sample["label"].astype(int)
    categories = sample["attack_cat"]

    # Fit / calibration / internal validation = 70% / 15% / 15%.
    X_fit, X_rest, y_fit, y_rest, cat_fit, cat_rest = train_test_split(
        X, y, categories, test_size=0.30,
        random_state=SEED, stratify=y
    )
    X_cal, X_val, y_cal, y_val, cat_cal, cat_val = train_test_split(
        X_rest, y_rest, cat_rest, test_size=0.50,
        random_state=SEED + 1, stratify=y_rest
    )

    model, features = make_model(X_fit)
    model.fit(X_fit[features], y_fit)

    # Threshold calibration uses only normal calibration records.
    normal_cal = (cat_cal == "Normal").to_numpy()
    if not normal_cal.any():
        raise ValueError("No normal records in calibration split.")

    cal_scores = model.predict_proba(X_cal[features])[:, 1]
    threshold = threshold_for_fpr(cal_scores[normal_cal], TARGET_FPR)

    # Internal validation: known categories only.
    val_scores = model.predict_proba(X_val[features])[:, 1]
    val_pred = (val_scores >= threshold).astype(int)
    normal_val = (cat_val == "Normal").to_numpy()
    val_fp = int(((val_pred == 1) & normal_val).sum())
    val_normal_count = int(normal_val.sum())
    val_fpr = val_fp / val_normal_count if val_normal_count else None

    # Separate test data: balanced normal vs excluded attack category.
    normal_test = test_df[test_df["attack_cat"] == "Normal"]
    unseen_test = test_df[test_df["attack_cat"] == held_out]

    if len(normal_test) < TEST_PER_CLASS or len(unseen_test) < TEST_PER_CLASS:
        raise ValueError(
            f"Insufficient test records for {held_out}: "
            f"normal={len(normal_test)}, held_out={len(unseen_test)}"
        )

    normal_test = normal_test.sample(n=TEST_PER_CLASS, random_state=SEED)
    unseen_test = unseen_test.sample(n=TEST_PER_CLASS, random_state=SEED)

    eval_df = pd.concat([normal_test, unseen_test], ignore_index=True)
    eval_df = eval_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_test = eval_df[features].copy()
    y_test = eval_df["label"].astype(int).to_numpy()

    # Confirm this is a binary normal-vs-held-out attack evaluation.
    assert set(y_test) == {0, 1}
    assert held_out not in set(sample["attack_cat"])
    assert set(eval_df["attack_cat"]) == {"Normal", held_out}

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    test_normal_mask = (eval_df["attack_cat"] == "Normal").to_numpy()
    test_attack_mask = (eval_df["attack_cat"] == held_out).to_numpy()

    attack_count = int(test_attack_mask.sum())
    normal_count = int(test_normal_mask.sum())
    attack_recall = tp / attack_count if attack_count else None
    test_fpr = fp / normal_count if normal_count else None

    result = {
        "run_id": RUN_ID,
        "held_out_category": held_out,
        "model": "RandomForestClassifier",
        "seed": SEED,
        "training_sample_rows": int(len(sample)),
        "fit_rows": int(len(X_fit)),
        "calibration_rows": int(len(X_cal)),
        "internal_validation_rows": int(len(X_val)),
        "held_out_category_in_training_partitions": False,
        "target_calibration_fpr": TARGET_FPR,
        "calibration_normal_rows": int(normal_cal.sum()),
        "selected_threshold": float(threshold),
        "internal_validation_known_attack_recall": float(
            recall_score(y_val, val_pred, zero_division=0)
        ),
        "internal_validation_fpr": val_fpr,
        "test_normal_rows": normal_count,
        "test_held_out_attack_rows": attack_count,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_held_out_attack_recall": float(attack_recall),
        "test_fpr": float(test_fpr),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "elapsed_seconds": round(time.time() - start, 2),
        "interpretation": (
            "Exploratory withheld-category evaluation on UNSW-NB15. "
            "The testing data has been inspected in earlier experiments; "
            "this is not an untouched final test or proof of real-world "
            "zero-day detection."
        ),
    }

    all_results.append(result)
    joblib.dump(model, MODEL_DIR / f"{held_out.lower()}_random_forest.joblib")

    print(f"Training rows: {len(X_fit)}")
    print(f"Calibration normal rows: {int(normal_cal.sum())}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Test records: normal={normal_count}, {held_out}={attack_count}")
    print(f"Held-out attack recall: {attack_recall:.4f}")
    print(f"False-positive rate: {test_fpr:.4f}")
    print(f"ROC-AUC: {result['test_roc_auc']:.4f}")
    print(f"Confusion matrix: {result['confusion_matrix']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']}")

csv_path = OUT_DIR / "rf_calibrated_heldout_results.csv"
json_path = OUT_DIR / "rf_calibrated_heldout_results.json"

pd.DataFrame(all_results).to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump({
        "experiment": "random_forest_calibrated_heldout_category",
        "run_id": RUN_ID,
        "dataset_training_file": str(TRAIN_FILE),
        "dataset_testing_file": str(TEST_FILE),
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "train_sample_rows": TRAIN_SAMPLE,
        "test_per_class": TEST_PER_CLASS,
        "target_calibration_fpr": TARGET_FPR,
        "results": all_results,
        "limitations": [
            "Threshold is selected using normal calibration records only.",
            "Held-out attack category is excluded from training-side partitions.",
            "The testing dataset was inspected in earlier experiments.",
            "Results are exploratory and do not prove real-world zero-day detection.",
        ],
    }, f, indent=2)

print("\n=== SUMMARY ===")
summary_cols = [
    "held_out_category", "selected_threshold",
    "test_held_out_attack_recall", "test_fpr",
    "test_precision", "test_f1", "test_roc_auc",
    "test_average_precision", "false_positives", "false_negatives"
]
print(pd.DataFrame(all_results)[summary_cols].round(4).to_string(index=False))
print(f"\nCSV: {csv_path}")
print(f"JSON: {json_path}")
print(f"Models: {MODEL_DIR}")
