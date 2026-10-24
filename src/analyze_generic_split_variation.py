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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")
SEEDS = [7, 21, 42]
TARGET_FPR = 0.05
TRAIN_SAMPLE_SIZE = 20000
NORMAL_EVAL_SIZE = 5000
ATTACK_EVAL_SIZE = 5000

TOP5_GENERIC = ["sttl", "proto", "dttl", "ct_dst_src_ltm", "dbytes"]

RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = Path("reports/generic_split_variation") / RUN_ID
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

if len(generic_all) < ATTACK_EVAL_SIZE:
    raise ValueError(f"Not enough Generic rows: {len(generic_all)}")

all_scores = []
all_score_summaries = []
all_feature_summaries = []

feature_cols = [
    c for c in df.columns if c not in {"id", "attack_cat", "label"}
]
numeric_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

def build_model(features):
    selected_numeric_cols = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    selected_categorical_cols = [
        c for c in features if c not in selected_numeric_cols
    ]

    transformers = []

    if selected_numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            selected_numeric_cols,
        ))

    if selected_categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            selected_categorical_cols,
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

def standardized_mean_difference(a, b):
    a = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan)
    b = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan)

    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()

    if pd.isna(ma) or pd.isna(mb):
        return np.nan

    pooled = np.sqrt((sa**2 + sb**2) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if ma == mb else np.nan

    return float((ma - mb) / pooled)

# Select evaluation rows ONCE so every seed uses identical records.
FIXED_NORMAL_EVAL_SEED = 20261024
FIXED_GENERIC_EVAL_SEED = 20261025

normal_eval = normal_all.sample(
    n=NORMAL_EVAL_SIZE,
    random_state=FIXED_NORMAL_EVAL_SEED,
)
generic_eval = generic_all.sample(
    n=ATTACK_EVAL_SIZE,
    random_state=FIXED_GENERIC_EVAL_SEED,
)
remaining_normal = normal_all.drop(index=normal_eval.index)

print(
    f"Fixed evaluation cohort: {len(normal_eval)} normal rows and "
    f"{len(generic_eval)} held-out Generic rows"
)
print(
    f"Evaluation selection seeds: normal={FIXED_NORMAL_EVAL_SEED}, "
    f"Generic={FIXED_GENERIC_EVAL_SEED}"
)

for seed in SEEDS:
    print(f"\n=== Fixed training pool; split seed {seed} ===")

    # Evaluation rows remain fixed; only the training pool and split vary.
    pool = pd.concat([remaining_normal, known_attacks], axis=0)
    pool = pool.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(pool)),
        random_state=20261026,
    )

    fit_df, cal_df = train_test_split(
        pool,
        test_size=0.30,
        random_state=seed,
        stratify=pool["label"],
    )

    eval_df = pd.concat(
        [normal_eval, generic_eval], axis=0
    ).copy()

    eval_df["eval_group"] = np.where(
        eval_df["label"].to_numpy() == 0,
        "Normal",
        "Heldout_Generic",
    )

    variants = {
        "full": feature_cols,
        "no_top5": [c for c in feature_cols if c not in TOP5_GENERIC],
    }

    for variant, features in variants.items():
        print(f"  Fitting {variant} ...")

        model = build_model(features)
        model.fit(fit_df[features], fit_df["label"])

        cal_normal = cal_df[cal_df["label"] == 0]
        p_cal_normal = model.predict_proba(
            cal_normal[features]
        )[:, 1]

        threshold = float(np.quantile(
            p_cal_normal, 1 - TARGET_FPR
        ))

        p_eval = model.predict_proba(eval_df[features])[:, 1]
        y_true = eval_df["label"].to_numpy()
        y_pred = (p_eval >= threshold).astype(int)

        tn, fp, fn_count, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        scores = eval_df[
            ["label", "eval_group"] + (
                ["attack_cat"] if "attack_cat" in eval_df.columns else []
            )
        ].copy()
        scores["seed"] = seed
        scores["variant"] = variant
        scores["score"] = p_eval
        scores["threshold"] = threshold
        scores["score_minus_threshold"] = p_eval - threshold
        scores["predicted_label"] = y_pred
        scores["error_type"] = np.select(
            [
                (y_true == 0) & (y_pred == 0),
                (y_true == 0) & (y_pred == 1),
                (y_true == 1) & (y_pred == 0),
                (y_true == 1) & (y_pred == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="Unknown",
        )

        # Preserve original feature values for subsequent FN/TP analysis.
        for col in numeric_cols:
            scores[col] = eval_df[col].to_numpy()

        all_scores.append(scores)

        # Score distributions for normal traffic and held-out Generic attacks.
        for group in ["Normal", "Heldout_Generic"]:
            subset = scores[scores["eval_group"] == group]
            s = subset["score"]
            margin = subset["score_minus_threshold"]

            all_score_summaries.append({
                "seed": seed,
                "variant": variant,
                "group": group,
                "n": len(subset),
                "threshold": threshold,
                "score_mean": s.mean(),
                "score_std": s.std(),
                "score_p05": s.quantile(0.05),
                "score_p25": s.quantile(0.25),
                "score_median": s.median(),
                "score_p75": s.quantile(0.75),
                "score_p95": s.quantile(0.95),
                "below_threshold_fraction": (s < threshold).mean(),
                "margin_p05": margin.quantile(0.05),
                "margin_median": margin.median(),
                "margin_p95": margin.quantile(0.95),
            })

        # Attack score bands show whether misses are near or far below threshold.
        attacks = scores[scores["eval_group"] == "Heldout_Generic"]
        bands = [
            ("far_below", lambda m: m < -0.20),
            ("below_0.20", lambda m: (m >= -0.20) & (m < -0.10)),
            ("below_0.10", lambda m: (m >= -0.10) & (m < 0)),
            ("above_0.00_to_0.10", lambda m: (m >= 0) & (m < 0.10)),
            ("above_0.10_to_0.20", lambda m: (m >= 0.10) & (m < 0.20)),
            ("far_above", lambda m: m >= 0.20),
        ]

        for band_name, band_condition in bands:
            margin = attacks["score_minus_threshold"]
            all_score_summaries.append({
                "seed": seed,
                "variant": variant,
                "group": f"Generic_margin_band:{band_name}",
                "n": int(band_condition(margin).sum()),
                "threshold": threshold,
                "score_mean": attacks.loc[band_condition(margin), "score"].mean(),
                "score_std": attacks.loc[band_condition(margin), "score"].std(),
                "score_p05": np.nan,
                "score_p25": np.nan,
                "score_median": attacks.loc[band_condition(margin), "score"].median(),
                "score_p75": np.nan,
                "score_p95": np.nan,
                "below_threshold_fraction": np.nan,
                "margin_p05": np.nan,
                "margin_median": attacks.loc[band_condition(margin), "score_minus_threshold"].median(),
                "margin_p95": np.nan,
            })

        # Compare numeric features of missed vs detected Generic attacks.
        fn_rows = scores[
            (scores["eval_group"] == "Heldout_Generic") &
            (scores["error_type"] == "FN")
        ]
        tp_rows = scores[
            (scores["eval_group"] == "Heldout_Generic") &
            (scores["error_type"] == "TP")
        ]

        for col in numeric_cols:
            fn_values = fn_rows[col]
            tp_values = tp_rows[col]

            all_feature_summaries.append({
                "seed": seed,
                "variant": variant,
                "feature": col,
                "fn_count": len(fn_rows),
                "tp_count": len(tp_rows),
                "fn_mean": fn_values.mean(),
                "tp_mean": tp_values.mean(),
                "fn_median": fn_values.median(),
                "tp_median": tp_values.median(),
                "fn_zero_fraction": (fn_values == 0).mean(),
                "tp_zero_fraction": (tp_values == 0).mean(),
                "smd_fn_minus_tp": standardized_mean_difference(
                    fn_values, tp_values
                ),
            })

        print(
            f"    threshold={threshold:.4f}, "
            f"Generic recall={tp / (tp + fn_count):.4f}, "
            f"normal FPR={fp / (fp + tn):.4f}, "
            f"ROC-AUC={roc_auc_score(y_true, p_eval):.4f}, "
            f"FN={fn_count}, TP={tp}"
        )

scores_df = pd.concat(all_scores, ignore_index=True)
summary_df = pd.DataFrame(all_score_summaries)
features_df = pd.DataFrame(all_feature_summaries)

scores_df.to_csv(OUT / "generic_row_scores.csv", index=False)
summary_df.to_csv(OUT / "generic_score_distributions.csv", index=False)
features_df.to_csv(OUT / "generic_fn_vs_tp_features.csv", index=False)

# Rank features by absolute FN-vs-TP standardized difference for each run.
ranked = features_df.copy()
ranked["abs_smd"] = ranked["smd_fn_minus_tp"].abs()
ranked = ranked.sort_values(
    ["seed", "variant", "abs_smd"], ascending=[True, True, False]
)
ranked.to_csv(OUT / "generic_fn_vs_tp_features_ranked.csv", index=False)

metadata = {
    "run_id": RUN_ID,
    "dataset": str(DATA),
    "dataset_scope": "training CSV only",
    "heldout_category": "Generic",
    "seeds": SEEDS,
    "experiment": "fixed_training_pool_vary_fit_calibration_split",
    "fixed_training_pool_seed": 20261026,
    "variants": ["full", "no_top5"],
    "normal_eval_size": NORMAL_EVAL_SIZE,
    "heldout_attack_eval_size": ATTACK_EVAL_SIZE,
    "threshold_target_fpr": TARGET_FPR,
    "threshold_source": "normal calibration rows only",
    "feature_comparison": "descriptive numeric feature differences between false negatives and true positives",
    "interpretation_limit": (
        "Dataset-specific exploratory diagnostics; feature differences "
        "are associations, not causal explanations or proof of real-world "
        "zero-day detection."
    ),
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\n=== Score distributions ===")
print(summary_df[
    summary_df["group"].isin(["Normal", "Heldout_Generic"])
][[
    "seed", "variant", "group", "n", "threshold",
    "score_median", "score_p05", "score_p95",
    "below_threshold_fraction",
]].to_string(index=False))

print("\n=== Top FN-vs-TP feature differences per seed/variant ===")
for (seed, variant), group in ranked.groupby(["seed", "variant"]):
    print(f"\nSeed={seed}, variant={variant}")
    print(group.head(8)[[
        "feature", "fn_mean", "tp_mean",
        "fn_median", "tp_median", "smd_fn_minus_tp"
    ]].to_string(index=False))

print(f"\nSaved all outputs to: {OUT}")
