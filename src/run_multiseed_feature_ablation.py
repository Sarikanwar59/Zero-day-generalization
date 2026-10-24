from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import time

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

TRAIN_FILE = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")
SEEDS = [7, 21, 42]
SAMPLE_SIZE = 20000
TARGET_CALIBRATION_FPR = 0.05

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path("reports/feature_ablation_multiseed") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=False)

# Feature groups are fixed from the prior seed-42 importance analysis.
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


def make_pipeline(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

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

    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([("preprocess", prep), ("model", rf)])


def get_threshold(y_cal, scores_cal):
    normal_scores = scores_cal[y_cal.to_numpy() == 0]
    if len(normal_scores) == 0:
        raise ValueError("No normal calibration rows.")
    return float(pd.Series(normal_scores).quantile(1.0 - TARGET_CALIBRATION_FPR))


def run_variant(held_out, variant, drop_features, seed,
                X_fit, y_fit, X_cal, y_cal, X_val, y_val):
    drop = [f for f in drop_features if f in X_fit.columns]
    columns = [c for c in X_fit.columns if c not in drop]
    model = make_pipeline(X_fit[columns])
    model.fit(X_fit[columns], y_fit)

    cal_scores = model.predict_proba(X_cal[columns])[:, 1]
    threshold = get_threshold(y_cal, cal_scores)

    val_scores = model.predict_proba(X_val[columns])[:, 1]
    pred = (val_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()

    result = {
        "seed": seed,
        "held_out_category": held_out,
        "variant": variant,
        "dropped_features": ";".join(drop),
        "dropped_feature_count": len(drop),
        "remaining_feature_count": len(columns),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "attack_recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "roc_auc": float(roc_auc_score(y_val, val_scores)),
        "average_precision": float(average_precision_score(y_val, val_scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_rows": len(X_fit),
        "calibration_rows": len(X_cal),
        "validation_rows": len(X_val),
    }
    del model
    gc.collect()
    return result


def main():
    start = time.time()
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    print(f"Reading training CSV only: {TRAIN_FILE}", flush=True)
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    if not {"label", "attack_cat"}.issubset(df.columns):
        raise ValueError("Expected label and attack_cat columns.")

    all_results = []

    for held_out in ["Generic", "Exploits"]:
        for seed in SEEDS:
            print(f"\n=== Excluding {held_out} | seed={seed} ===", flush=True)

            pool = df[df["attack_cat"] != held_out].copy()
            if len(pool) > SAMPLE_SIZE:
                pool = pool.sample(n=SAMPLE_SIZE, random_state=seed)

            y = pool["label"].astype(int)
            X = pool.drop(columns=["id", "attack_cat", "label"], errors="ignore")

            X_fit, X_temp, y_fit, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=seed, stratify=y
            )
            X_cal, X_val, y_cal, y_val = train_test_split(
                X_temp, y_temp, test_size=0.50,
                random_state=seed + 1, stratify=y_temp
            )

            variants = {"full": [], **ABLATIONS[held_out]}
            for variant, dropped in variants.items():
                print(f"Running {variant}...", flush=True)
                result = run_variant(
                    held_out, variant, dropped, seed,
                    X_fit, y_fit, X_cal, y_cal, X_val, y_val
                )
                all_results.append(result)
                print(
                    f"  AUC={result['roc_auc']:.4f} "
                    f"F1={result['f1']:.4f} "
                    f"Recall={result['attack_recall']:.4f} "
                    f"FPR={result['fpr']:.4f}",
                    flush=True
                )

            del pool, X, y, X_fit, X_temp, y_fit, y_temp
            del X_cal, X_val, y_cal, y_val
            gc.collect()

    raw = pd.DataFrame(all_results)
    raw_path = OUT_DIR / "feature_ablation_by_seed.csv"
    raw.to_csv(raw_path, index=False)

    metrics = [
        "accuracy", "precision", "attack_recall", "f1", "fpr",
        "roc_auc", "average_precision", "threshold"
    ]
    summary = (
        raw.groupby(["held_out_category", "variant"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary_path = OUT_DIR / "feature_ablation_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    # Paired per-seed differences against the full model for each category.
    paired_rows = []
    for held_out in ["Generic", "Exploits"]:
        subset = raw[raw["held_out_category"] == held_out]
        baseline = subset[subset["variant"] == "full"].set_index("seed")
        for variant in ["no_ttl", "no_service_protocol", "no_top5"]:
            ablated = subset[subset["variant"] == variant].set_index("seed")
            for seed in SEEDS:
                row = {
                    "held_out_category": held_out,
                    "variant": variant,
                    "seed": seed,
                }
                for metric in metrics:
                    row[f"{metric}_change_vs_full"] = (
                        ablated.loc[seed, metric] - baseline.loc[seed, metric]
                    )
                paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    paired_path = OUT_DIR / "feature_ablation_paired_deltas.csv"
    paired.to_csv(paired_path, index=False)

    metadata = {
        "experiment": "multi_seed_training_side_feature_ablation",
        "run_id": RUN_ID,
        "dataset": str(TRAIN_FILE),
        "testing_csv_loaded": False,
        "seeds": SEEDS,
        "sample_size_per_seed_and_category": SAMPLE_SIZE,
        "partition": "70% fit / 15% calibration / 15% validation",
        "target_calibration_fpr": TARGET_CALIBRATION_FPR,
        "random_forest_random_state": 42,
        "scikit_learn_version": sklearn.__version__,
        "runs": len(all_results),
        "notes": [
            "Seed changes sample, split, and validation rows.",
            "Random Forest internal random_state remains 42 for controlled comparisons.",
            "Feature groups were selected from a previous seed-42 importance run.",
            "This is training-side known-category validation, not held-out attack recall.",
            "Standard deviations are sample standard deviations across three seeds.",
            "No testing CSV was loaded.",
        ],
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== MEAN AND STANDARD DEVIATION ACROSS SEEDS ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== PAIRED ROC-AUC / F1 / RECALL CHANGES VS FULL ===")
    chosen = [
        "held_out_category", "variant", "seed",
        "roc_auc_change_vs_full", "f1_change_vs_full",
        "attack_recall_change_vs_full", "fpr_change_vs_full",
    ]
    print(paired[chosen].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print(f"\nSaved reports to: {OUT_DIR}")
    print(f"Raw per-seed results: {raw_path}")
    print(f"Mean/std summary: {summary_path}")
    print(f"Paired deltas: {paired_path}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")
    print("No testing CSV was loaded or evaluated.")


if __name__ == "__main__":
    main()
