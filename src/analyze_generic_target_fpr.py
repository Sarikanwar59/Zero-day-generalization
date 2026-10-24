import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Experiment configuration
# -----------------------------
DATA_PATH = "data/raw/unsw_nb15/UNSW_NB15_training-set.csv"

TRAIN_POOL_SEED = 20261026
FIT_SAMPLE_SEED = 20261027

NORMAL_EVAL_SEED = 20261024
GENERIC_EVAL_SEED = 20261025

TRAIN_SAMPLE_SIZE = 20000
NORMAL_EVAL_SIZE = 5000
GENERIC_EVAL_SIZE = 5000

CALIBRATION_SIZE = 3000
CALIBRATION_SEEDS = list(range(1101, 1111))
TARGET_FPRS = [0.01, 0.02, 0.05, 0.10]

# Generic-specific top five features from the earlier
# training-side permutation-importance experiment.
TOP5_GENERIC = [
    "sttl",
    "proto",
    "dttl",
    "ct_dst_src_ltm",
    "dbytes",
]

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = f"reports/generic_target_fpr/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)


def build_model(feature_columns):
    """Build a preprocessing + Random Forest pipeline."""
    numeric_features = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = [
        c for c in feature_columns
        if c not in numeric_features
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", classifier),
    ])


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


print(f"Run ID: {RUN_ID}")
print("Loading UNSW-NB15 training CSV...")
df = pd.read_csv(DATA_PATH, low_memory=False)

required = {"label", "attack_cat"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

# Remove rows with missing labels/categories and normalize category text.
df = df.dropna(subset=["label", "attack_cat"]).copy()
df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

normal_indices = df.index[df["label"].astype(int) == 0].to_numpy()
generic_indices = df.index[
    (df["label"].astype(int) == 1)
    & (df["attack_cat"].str.lower() == "generic")
].to_numpy()

if len(normal_indices) < NORMAL_EVAL_SIZE + CALIBRATION_SIZE:
    raise ValueError("Not enough normal rows for evaluation and calibration.")
if len(generic_indices) < GENERIC_EVAL_SIZE:
    raise ValueError("Not enough Generic attack rows for evaluation.")

# Fixed evaluation cohort, selected once.
rng_normal = np.random.default_rng(NORMAL_EVAL_SEED)
eval_normal_indices = rng_normal.choice(
    normal_indices, size=NORMAL_EVAL_SIZE, replace=False
)

rng_generic = np.random.default_rng(GENERIC_EVAL_SEED)
eval_generic_indices = rng_generic.choice(
    generic_indices, size=GENERIC_EVAL_SIZE, replace=False
)

eval_indices = set(eval_normal_indices.tolist()) | set(
    eval_generic_indices.tolist()
)

eval_df = df.loc[
    np.concatenate([eval_normal_indices, eval_generic_indices])
].copy()
y_eval = eval_df["label"].astype(int).to_numpy()

# Training pool excludes Generic attacks and all evaluation rows.
train_pool = df[
    (df["attack_cat"].str.lower() != "generic")
    & (~df.index.isin(eval_indices))
].copy()

if len(train_pool) < TRAIN_SAMPLE_SIZE:
    raise ValueError(
        f"Training pool has {len(train_pool)} rows, "
        f"fewer than {TRAIN_SAMPLE_SIZE} required."
    )

fit_rows = train_pool.sample(
    n=TRAIN_SAMPLE_SIZE,
    random_state=FIT_SAMPLE_SEED,
).copy()
fit_indices = set(fit_rows.index.tolist())

# Calibration reservoir: normal rows not used for fit or evaluation.
calibration_reservoir = df[
    (df["label"].astype(int) == 0)
    & (~df.index.isin(eval_indices))
    & (~df.index.isin(fit_indices))
].copy()

if len(calibration_reservoir) < CALIBRATION_SIZE:
    raise ValueError(
        f"Calibration reservoir has only {len(calibration_reservoir)} rows."
    )

excluded_columns = {"id", "attack_cat", "label"}
all_features = [
    c for c in df.columns if c not in excluded_columns
]
no_top5_features = [
    c for c in all_features if c not in TOP5_GENERIC
]

variants = {
    "full": all_features,
    "no_top5": no_top5_features,
}

print(f"Rows loaded: {len(df):,}")
print(f"Training pool: {len(train_pool):,}")
print(f"Fixed fit rows: {len(fit_rows):,}")
print(f"Calibration reservoir: {len(calibration_reservoir):,}")
print(f"Evaluation rows: {len(eval_df):,}")
print(f"Evaluation normals: {(y_eval == 0).sum():,}")
print(f"Evaluation Generic attacks: {(y_eval == 1).sum():,}")
print(f"Calibration sample size per replicate: {CALIBRATION_SIZE:,}")
print(f"Targets: {TARGET_FPRS}")
print(f"Calibration seeds: {CALIBRATION_SEEDS}")

results = []

for variant, features in variants.items():
    print(f"\n=== Fitting variant: {variant} ===")

    model = build_model(features)
    model.fit(fit_rows[features], fit_rows["label"].astype(int))

    # Fixed evaluation scores for this fitted model.
    eval_scores = model.predict_proba(eval_df[features])[:, 1]

    # Score the normal reservoir once; later samples select from these scores.
    reservoir_scores = model.predict_proba(
        calibration_reservoir[features]
    )[:, 1]

    # Use the same calibration sample for all target FPRs in each replicate.
    for seed in CALIBRATION_SEEDS:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(reservoir_scores),
            size=CALIBRATION_SIZE,
            replace=False,
        )
        calibration_scores = reservoir_scores[positions]

        for target_fpr in TARGET_FPRS:
            # Empirical calibration quantile. Ties in scores can cause the
            # realized calibration FPR to differ slightly from the target.
            threshold = float(
                np.quantile(calibration_scores, 1.0 - target_fpr)
            )

            metrics = calculate_metrics(y_eval, eval_scores, threshold)
            calibration_fpr = float(
                np.mean(calibration_scores >= threshold)
            )

            results.append({
                "variant": variant,
                "target_fpr": target_fpr,
                "calibration_seed": seed,
                "calibration_size": CALIBRATION_SIZE,
                "calibration_fpr": calibration_fpr,
                "evaluation_rows": len(y_eval),
                **metrics,
            })

    del model

results_df = pd.DataFrame(results)
results_path = os.path.join(OUT_DIR, "target_fpr_results.csv")
results_df.to_csv(results_path, index=False)

summary_rows = []
group_columns = ["variant", "target_fpr"]

for (variant, target_fpr), group in results_df.groupby(group_columns):
    row = {
        "variant": variant,
        "target_fpr": float(target_fpr),
        "replicates": int(len(group)),
    }

    for metric in [
        "threshold",
        "calibration_fpr",
        "recall",
        "fpr",
        "precision",
        "f1",
    ]:
        values = group[metric].astype(float)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_sd"] = float(values.std(ddof=1))
        row[f"{metric}_min"] = float(values.min())
        row[f"{metric}_max"] = float(values.max())
        if metric in ["recall", "fpr"]:
            row[f"{metric}_range_percentage_points"] = float(
                (values.max() - values.min()) * 100
            )

    # Ranking metrics are fixed for a given fitted model and eval cohort.
    row["roc_auc"] = float(group["roc_auc"].iloc[0])
    row["average_precision"] = float(group["average_precision"].iloc[0])
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "target_fpr_summary.csv")
summary_df.to_csv(summary_path, index=False)

metadata = {
    "run_id": RUN_ID,
    "dataset_scope": "UNSW-NB15 training CSV only",
    "heldout_category": "Generic",
    "target_fprs": TARGET_FPRS,
    "calibration_size": CALIBRATION_SIZE,
    "calibration_seeds": CALIBRATION_SEEDS,
    "training_pool_seed": TRAIN_POOL_SEED,
    "fit_sample_seed": FIT_SAMPLE_SEED,
    "normal_eval_seed": NORMAL_EVAL_SEED,
    "generic_eval_seed": GENERIC_EVAL_SEED,
    "fit_sample_size": TRAIN_SAMPLE_SIZE,
    "normal_eval_size": NORMAL_EVAL_SIZE,
    "generic_eval_size": GENERIC_EVAL_SIZE,
    "calibration_reservoir_size": int(len(calibration_reservoir)),
    "variants": list(variants.keys()),
    "no_top5_features_removed": TOP5_GENERIC,
    "threshold_rule": "quantile(calibration_normal_scores, 1 - target_fpr)",
    "notes": [
        "Model fit rows and evaluation rows are fixed within this run.",
        "Each calibration replicate is reused across all target FPRs.",
        "Calibration samples can overlap across replicates.",
        "Calibration FPR may differ slightly from target due to quantile "
        "interpolation and tied scores.",
        "Evaluation cohort is balanced and is not a real-world prevalence "
        "estimate.",
        "Exploratory held-out-category evaluation is not proof of real-world "
        "zero-day detection.",
    ],
}

metadata_path = os.path.join(OUT_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))
print(f"\nSaved detailed results: {results_path}")
print(f"Saved summary: {summary_path}")
print(f"Saved metadata: {metadata_path}")
