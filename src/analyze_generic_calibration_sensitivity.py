import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")

TARGET_FPR = 0.05
TRAIN_SAMPLE_SIZE = 20000
NORMAL_EVAL_SIZE = 5000
ATTACK_EVAL_SIZE = 5000

# Fixed choices: only the calibration sample will vary.
TRAIN_POOL_SEED = 20261026
FIT_SAMPLE_SEED = 20261027
NORMAL_EVAL_SEED = 20261024
GENERIC_EVAL_SEED = 20261025

CALIBRATION_SEEDS = [101, 202, 303, 404, 505]
CALIBRATION_SAMPLE_SIZE = 3000

TOP5_GENERIC = ["sttl", "proto", "dttl", "ct_dst_src_ltm", "dbytes"]

RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = Path("reports/generic_calibration_sensitivity") / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Reading only: {DATA}")

df = pd.read_csv(DATA, low_memory=False)
df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

normal_all = df[df["label"] == 0].copy()
generic_all = df[
    (df["label"] == 1) & (df["attack_cat"] == "Generic")
].copy()
known_attacks = df[
    (df["label"] == 1) & (df["attack_cat"] != "Generic")
].copy()

if len(normal_all) < NORMAL_EVAL_SIZE + CALIBRATION_SAMPLE_SIZE:
    raise ValueError("Not enough normal rows for evaluation and calibration.")
if len(generic_all) < ATTACK_EVAL_SIZE:
    raise ValueError(f"Not enough Generic rows: {len(generic_all)}")

feature_cols = [
    c for c in df.columns if c not in {"id", "attack_cat", "label"}
]

def build_model(features):
    numeric_cols = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in features if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    preprocess = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )),
    ])

# Fixed evaluation cohort, selected once.
normal_eval = normal_all.sample(
    n=NORMAL_EVAL_SIZE,
    random_state=NORMAL_EVAL_SEED,
)
generic_eval = generic_all.sample(
    n=ATTACK_EVAL_SIZE,
    random_state=GENERIC_EVAL_SEED,
)
eval_df = pd.concat([normal_eval, generic_eval], axis=0).copy()
y_eval = eval_df["label"].to_numpy()

print(
    f"Fixed evaluation cohort: {len(normal_eval)} normal, "
    f"{len(generic_eval)} Generic attack rows"
)

# Build a fixed training pool excluding held-out Generic attacks and
# evaluation rows. Fit on this same pool for every calibration replicate.
remaining_normal = normal_all.drop(index=normal_eval.index)
pool = pd.concat([remaining_normal, known_attacks], axis=0)
pool = pool.sample(
    n=min(TRAIN_SAMPLE_SIZE, len(pool)),
    random_state=TRAIN_POOL_SEED,
)

fit_df = pool.sample(
    frac=1.0,
    random_state=FIT_SAMPLE_SEED,
).copy()

# Calibration reservoir contains only normal records not used for fitting
# or evaluation. Remove any fit-pool normal rows by their original index.
calibration_reservoir = remaining_normal.drop(
    index=fit_df.index.intersection(remaining_normal.index),
)

if len(calibration_reservoir) < CALIBRATION_SAMPLE_SIZE:
    raise ValueError(
        f"Calibration reservoir too small: {len(calibration_reservoir)}"
    )

print(f"Fixed fit rows: {len(fit_df)}")
print(f"Normal calibration reservoir: {len(calibration_reservoir)}")
print(f"Calibration sample size: {CALIBRATION_SAMPLE_SIZE}")
print(f"Calibration seeds: {CALIBRATION_SEEDS}")

variants = {
    "full": feature_cols,
    "no_top5": [c for c in feature_cols if c not in TOP5_GENERIC],
}

results = []

for variant, features in variants.items():
    print(f"\n=== Fitting fixed model: {variant} ===")

    model = build_model(features)
    model.fit(fit_df[features], fit_df["label"])

    # Fixed evaluation scores: never refit this model within the variant.
    p_eval = model.predict_proba(eval_df[features])[:, 1]
    auc = float(roc_auc_score(y_eval, p_eval))
    ap = float(average_precision_score(y_eval, p_eval))

    print(f"Fixed evaluation ROC-AUC={auc:.5f}, AP={ap:.5f}")

    for cal_seed in CALIBRATION_SEEDS:
        cal_normal = calibration_reservoir.sample(
            n=CALIBRATION_SAMPLE_SIZE,
            random_state=cal_seed,
        )

        p_cal = model.predict_proba(cal_normal[features])[:, 1]
        threshold = float(np.quantile(p_cal, 1 - TARGET_FPR))

        y_pred = (p_eval >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_eval, y_pred, labels=[0, 1]
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else np.nan
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        row = {
            "variant": variant,
            "calibration_seed": cal_seed,
            "fit_rows": len(fit_df),
            "calibration_rows": len(cal_normal),
            "threshold": threshold,
            "target_calibration_fpr": TARGET_FPR,
            "evaluation_recall": recall,
            "evaluation_fpr": fpr,
            "evaluation_precision_balanced_sample": precision,
            "evaluation_f1_balanced_sample": f1,
            "roc_auc_fixed_scores": auc,
            "average_precision_fixed_scores": ap,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        results.append(row)

        print(
            f"  cal_seed={cal_seed}, threshold={threshold:.5f}, "
            f"recall={recall:.4f}, FPR={fpr:.4f}, "
            f"FN={fn}, FP={fp}"
        )

results_df = pd.DataFrame(results)
results_path = OUT / "calibration_sensitivity_results.csv"
results_df.to_csv(results_path, index=False)

summary_rows = []
for variant, group in results_df.groupby("variant"):
    summary_rows.append({
        "variant": variant,
        "replicates": len(group),
        "threshold_mean": group["threshold"].mean(),
        "threshold_sd": group["threshold"].std(ddof=1),
        "threshold_min": group["threshold"].min(),
        "threshold_max": group["threshold"].max(),
        "recall_mean": group["evaluation_recall"].mean(),
        "recall_sd": group["evaluation_recall"].std(ddof=1),
        "recall_min": group["evaluation_recall"].min(),
        "recall_max": group["evaluation_recall"].max(),
        "recall_range_percentage_points": (
            group["evaluation_recall"].max()
            - group["evaluation_recall"].min()
        ) * 100,
        "fpr_mean": group["evaluation_fpr"].mean(),
        "fpr_sd": group["evaluation_fpr"].std(ddof=1),
        "auc_min": group["roc_auc_fixed_scores"].min(),
        "auc_max": group["roc_auc_fixed_scores"].max(),
        "ap_min": group["average_precision_fixed_scores"].min(),
        "ap_max": group["average_precision_fixed_scores"].max(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT / "calibration_sensitivity_summary.csv"
summary_df.to_csv(summary_path, index=False)

metadata = {
    "run_id": RUN_ID,
    "dataset": str(DATA),
    "dataset_scope": "UNSW-NB15 training CSV only",
    "heldout_category": "Generic",
    "fit_pool_seed": TRAIN_POOL_SEED,
    "fit_sample_seed": FIT_SAMPLE_SEED,
    "fixed_normal_eval_seed": NORMAL_EVAL_SEED,
    "fixed_generic_eval_seed": GENERIC_EVAL_SEED,
    "calibration_seeds": CALIBRATION_SEEDS,
    "calibration_sample_size": CALIBRATION_SAMPLE_SIZE,
    "fit_rows": len(fit_df),
    "calibration_reservoir_rows": len(calibration_reservoir),
    "normal_eval_rows": len(normal_eval),
    "generic_eval_rows": len(generic_eval),
    "threshold_target_fpr": TARGET_FPR,
    "variants": ["full", "no_top5"],
    "design": (
        "One fixed fitted model per feature variant and one fixed evaluation "
        "cohort per variant. Only the normal calibration sample changes."
    ),
    "interpretation_limit": (
        "Exploratory dataset-specific calibration sensitivity experiment. "
        "Evaluation precision and F1 use a balanced 50:50 normal/attack cohort "
        "and are not deployment prevalence estimates. This is not proof of "
        "real-world zero-day detection."
    ),
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\n=== Calibration sensitivity summary ===")
print(summary_df.to_string(index=False))
print(f"\nDetailed results: {results_path}")
print(f"Summary: {summary_path}")
print(f"Metadata: {OUT / 'metadata.json'}")
