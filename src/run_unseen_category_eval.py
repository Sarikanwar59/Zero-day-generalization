import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

DATA = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")
TARGET_FPR = 0.05
TRAIN_SAMPLE_SIZE = 20000
EVAL_NORMAL_SIZE = 5000
EVAL_ATTACK_SIZE = 5000
SEEDS = [7, 21, 42]

# Top five features from the earlier training-side importance analysis.
TOP5 = {
    "Generic": ["sttl", "proto", "dttl", "ct_dst_src_ltm", "dbytes"],
    "Exploits": ["service", "sttl", "sbytes", "smean", "sload"],
}

RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = Path("reports/heldout_category") / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Reading: {DATA}")
print(f"scikit-learn: {sklearn.__version__}")

df = pd.read_csv(DATA, low_memory=False)
required = {"label", "attack_cat"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

# Identify normal traffic by label, not by attack_cat spelling.
normal_all = df[df["label"] == 0].copy()
if len(normal_all) < EVAL_NORMAL_SIZE + 1000:
    raise ValueError("Not enough normal rows for a separate evaluation partition.")

all_results = []

def build_model(feature_columns, seed):
    numeric_cols = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

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

for heldout in ["Generic", "Exploits"]:
    print(f"\n=== Held-out category: {heldout} ===")

    heldout_attacks = df[
        (df["label"] == 1) & (df["attack_cat"] == heldout)
    ].copy()

    known_attacks = df[
        (df["label"] == 1) & (df["attack_cat"] != heldout)
    ].copy()

    if len(heldout_attacks) < EVAL_ATTACK_SIZE:
        raise ValueError(
            f"Only {len(heldout_attacks)} rows for {heldout}; "
            f"cannot sample {EVAL_ATTACK_SIZE}."
        )

    for seed in SEEDS:
        print(f"\n[{heldout}] seed={seed}")

        # Normal evaluation rows are removed before training-pool sampling.
        normal_eval = normal_all.sample(
            n=EVAL_NORMAL_SIZE, random_state=seed
        )
        remaining_normal = normal_all.drop(index=normal_eval.index)

        # Held-out attacks are never included in training or calibration.
        attack_eval = heldout_attacks.sample(
            n=EVAL_ATTACK_SIZE, random_state=seed
        )

        training_pool = pd.concat(
            [remaining_normal, known_attacks],
            ignore_index=False,
        )

        n_sample = min(TRAIN_SAMPLE_SIZE, len(training_pool))
        sampled_pool = training_pool.sample(
            n=n_sample, random_state=seed
        )

        fit_df, cal_df = train_test_split(
            sampled_pool,
            test_size=0.30,
            random_state=seed,
            stratify=sampled_pool["label"],
        )

        feature_columns = [
            c for c in df.columns if c not in {"id", "attack_cat", "label"}
        ]

        variants = {
            "full": feature_columns,
            "no_top5": [
                c for c in feature_columns if c not in TOP5[heldout]
            ],
        }

        # Evaluation combines disjoint normal traffic and excluded-category attacks.
        eval_df = pd.concat(
            [normal_eval, attack_eval], ignore_index=True
        )
        y_eval = eval_df["label"].to_numpy()

        for variant, features in variants.items():
            model = build_model(features, seed)

            X_fit = fit_df[features]
            y_fit = fit_df["label"].to_numpy()
            X_cal = cal_df[features]
            y_cal = cal_df["label"].to_numpy()
            X_eval = eval_df[features]

            model.fit(X_fit, y_fit)

            # Calibrate threshold on NORMAL calibration rows only.
            cal_normal_mask = (y_cal == 0)
            if cal_normal_mask.sum() == 0:
                raise ValueError("No normal calibration rows available.")

            p_cal_normal = model.predict_proba(
                X_cal.loc[cal_normal_mask]
            )[:, 1]

            threshold = float(np.quantile(p_cal_normal, 1 - TARGET_FPR))

            p_eval = model.predict_proba(X_eval)[:, 1]
            pred = (p_eval >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_eval, pred, labels=[0, 1]
            ).ravel()

            row = {
                "run_id": RUN_ID,
                "heldout_category": heldout,
                "seed": seed,
                "variant": variant,
                "train_pool_rows": int(len(sampled_pool)),
                "fit_rows": int(len(fit_df)),
                "calibration_rows": int(len(cal_df)),
                "calibration_normal_rows": int(cal_normal_mask.sum()),
                "normal_eval_rows": int(len(normal_eval)),
                "heldout_attack_eval_rows": int(len(attack_eval)),
                "target_calibration_fpr": TARGET_FPR,
                "threshold": threshold,
                "accuracy": accuracy_score(y_eval, pred),
                "precision": precision_score(
                    y_eval, pred, zero_division=0
                ),
                "heldout_attack_recall": recall_score(y_eval, pred),
                "f1": f1_score(y_eval, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_eval, p_eval),
                "average_precision": average_precision_score(y_eval, p_eval),
                "normal_fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            all_results.append(row)

            print(
                f"  {variant}: "
                f"unseen_attack_recall={row['heldout_attack_recall']:.4f}, "
                f"normal_FPR={row['normal_fpr']:.4f}, "
                f"ROC-AUC={row['roc_auc']:.4f}, "
                f"F1={row['f1']:.4f}, "
                f"threshold={threshold:.4f}"
            )

        # Save progress after each seed in case the session is interrupted.
        pd.DataFrame(all_results).to_csv(
            OUT / "unseen_category_by_seed.csv", index=False
        )

# Summarize means and sample standard deviations across the three seeds.
results = pd.DataFrame(all_results)
metrics = [
    "accuracy", "precision", "heldout_attack_recall", "f1",
    "normal_fpr", "roc_auc", "average_precision", "threshold",
]

summary = (
    results.groupby(["heldout_category", "variant"])[metrics]
    .agg(["mean", "std"])
)
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(OUT / "unseen_category_mean_std.csv", index=False)

metadata = {
    "run_id": RUN_ID,
    "dataset": str(DATA),
    "dataset_scope": "training CSV only",
    "heldout_categories": ["Generic", "Exploits"],
    "seeds": SEEDS,
    "train_sample_size": TRAIN_SAMPLE_SIZE,
    "normal_eval_size_per_seed": EVAL_NORMAL_SIZE,
    "heldout_attack_eval_size_per_seed": EVAL_ATTACK_SIZE,
    "target_calibration_fpr": TARGET_FPR,
    "threshold_calibration": "normal calibration rows only",
    "variants": ["full", "no_top5"],
    "random_forest_random_state": 42,
    "scikit_learn_version": sklearn.__version__,
    "interpretation_limit": (
        "Dataset-specific held-out-category evaluation; not proof of "
        "real-world zero-day detection. The excluded attack category is "
        "absent from model fitting and threshold calibration."
    ),
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\n=== Mean ± sample SD across seeds ===")
print(summary.to_string(index=False))
print(f"\nReports saved to: {OUT}")
