from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import time

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

TRAIN_FILE = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")
SEED = 42
SAMPLE_SIZE = 20000
TARGET_CALIBRATION_FPR = 0.05

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path("reports/feature_ablation_training_side") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=False)

# Fixed feature groups based on the previous training-side importance run.
# Rankings were obtained on the same validation partition, so these
# ablation results are exploratory, not an independent confirmation.
ABLATIONS = {
    "Generic": {
        "no_ttl": ["sttl", "dttl"],
        "no_service_protocol": ["service", "proto"],
        "no_top5": ["sttl", "proto", "dttl", "ct_dst_src_ltm", "dbytes"],
    },
    "Exploits": {
        "no_ttl": ["sttl", "dttl"],
        "no_service_protocol": ["service", "proto"],
        "no_top5": ["service", "sttl", "sbytes", "smean", "sload"],
    },
}


def make_pipeline(X_fit):
    categorical = X_fit.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X_fit.columns if c not in categorical]

    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def calibration_threshold(y_cal, scores_cal):
    # Choose the threshold using only normal calibration examples.
    normal_scores = np.asarray(scores_cal)[np.asarray(y_cal) == 0]
    if len(normal_scores) == 0:
        raise ValueError("No normal rows in calibration partition.")
    return float(np.quantile(normal_scores, 1.0 - TARGET_CALIBRATION_FPR))


def evaluate_variant(
    held_out,
    variant,
    drop_features,
    X_fit,
    y_fit,
    X_cal,
    y_cal,
    X_val,
    y_val,
):
    drop_features = [f for f in drop_features if f in X_fit.columns]
    fit_cols = [c for c in X_fit.columns if c not in drop_features]

    if not fit_cols:
        raise ValueError(f"No features left for {held_out}/{variant}")

    pipeline = make_pipeline(X_fit[fit_cols])
    pipeline.fit(X_fit[fit_cols], y_fit)

    cal_scores = pipeline.predict_proba(X_cal[fit_cols])[:, 1]
    threshold = calibration_threshold(y_cal, cal_scores)

    val_scores = pipeline.predict_proba(X_val[fit_cols])[:, 1]
    val_pred = (val_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")

    row = {
        "held_out_category": held_out,
        "variant": variant,
        "dropped_features": ";".join(drop_features),
        "dropped_feature_count": len(drop_features),
        "remaining_feature_count": len(fit_cols),
        "threshold_from_calibration": threshold,
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "validation_attack_recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "validation_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "validation_fpr": fpr,
        "validation_roc_auc": float(roc_auc_score(y_val, val_scores)),
        "validation_average_precision": float(average_precision_score(y_val, val_scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    del pipeline
    gc.collect()
    return row


def main():
    start = time.time()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_FILE}")

    print(f"Reading training CSV only: {TRAIN_FILE}", flush=True)
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    required = {"label", "attack_cat"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    all_rows = []

    for held_out in ["Generic", "Exploits"]:
        print(f"\n=== Feature ablation: excluding {held_out} ===", flush=True)

        pool = df[df["attack_cat"] != held_out].copy()
        if len(pool) > SAMPLE_SIZE:
            pool = pool.sample(n=SAMPLE_SIZE, random_state=SEED)

        y = pool["label"].astype(int)
        X = pool.drop(columns=["id", "attack_cat", "label"], errors="ignore")

        # Same 70/15/15 partition approach as the prior experiments.
        X_fit, X_temp, y_fit, y_temp = train_test_split(
            X, y,
            test_size=0.30,
            random_state=SEED,
            stratify=y,
        )
        X_cal, X_val, y_cal, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=43,
            stratify=y_temp,
        )

        print(
            f"Rows: fit={len(X_fit)}, calibration={len(X_cal)}, "
            f"validation={len(X_val)}",
            flush=True,
        )

        variants = {
            "full": [],
            **ABLATIONS[held_out],
        }

        for variant, drop_features in variants.items():
            print(f"Running {variant}...", flush=True)
            result = evaluate_variant(
                held_out,
                variant,
                drop_features,
                X_fit,
                y_fit,
                X_cal,
                y_cal,
                X_val,
                y_val,
            )
            all_rows.append(result)
            print(
                f"  ROC-AUC={result['validation_roc_auc']:.4f} | "
                f"F1={result['validation_f1']:.4f} | "
                f"Recall={result['validation_attack_recall']:.4f} | "
                f"FPR={result['validation_fpr']:.4f}",
                flush=True,
            )

        del pool, X, y, X_fit, X_temp, y_fit, y_temp
        del X_cal, X_val, y_cal, y_val
        gc.collect()

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "feature_ablation_results.csv", index=False)

    summary = {
        "experiment": "training_side_feature_ablation",
        "run_id": RUN_ID,
        "dataset": str(TRAIN_FILE),
        "testing_csv_loaded": False,
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "partition": "70% fit / 15% calibration / 15% validation",
        "target_calibration_fpr": TARGET_CALIBRATION_FPR,
        "results": all_rows,
        "limitations": [
            "Measures known-category validation performance, not held-out attack recall.",
            "Feature groups were selected using a previous importance run on the same validation split.",
            "Calibration thresholds are recalculated for each ablation using normal calibration rows.",
            "Results are exploratory and do not establish real-world zero-day detection.",
        ],
    }
    with open(OUT_DIR / "feature_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS ===")
    display_cols = [
        "held_out_category",
        "variant",
        "validation_roc_auc",
        "validation_precision",
        "validation_attack_recall",
        "validation_f1",
        "validation_fpr",
        "threshold_from_calibration",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved reports to: {OUT_DIR}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")
    print("No testing CSV was loaded or evaluated.")


if __name__ == "__main__":
    main()
