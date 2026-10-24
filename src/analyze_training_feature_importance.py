from pathlib import Path
from datetime import datetime, timezone
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

TRAIN_FILE = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")
SEED = 42
SAMPLE_SIZE = 20000
REPEATS = 3

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path("reports/feature_importance_training_side") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRAIN_FILE, low_memory=False)
df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)
df["label"] = pd.to_numeric(df["label"]).astype(int)

all_summaries = []

for held_out in ["Generic", "Exploits"]:
    print(f"\n=== Feature importance: excluding {held_out} ===")
    start = time.time()

    pool = df[df["attack_cat"] != held_out].copy()
    if len(pool) < SAMPLE_SIZE:
        raise ValueError(f"Not enough training rows after excluding {held_out}")

    sample = pool.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    excluded = {"id", "attack_cat", "label"}
    features = [c for c in sample.columns if c not in excluded]

    X = sample[features].copy()
    y = sample["label"].astype(int)
    categories = sample["attack_cat"]

    X_fit, X_rest, y_fit, y_rest, cat_fit, cat_rest = train_test_split(
        X, y, categories,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    X_cal, X_val, y_cal, y_val, cat_cal, cat_val = train_test_split(
        X_rest, y_rest, cat_rest,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=y_rest,
    )

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

    model.fit(X_fit[features], y_fit)

    # Internal validation contains known attack categories only.
    if held_out in set(cat_fit) | set(cat_cal) | set(cat_val):
        raise RuntimeError(f"Held-out category {held_out} entered a partition")

    val_scores = model.predict_proba(X_val[features])[:, 1]
    val_auc = roc_auc_score(y_val, val_scores)

    # 1. Permutation importance on raw features, using validation ROC-AUC.
    # Positive importance means shuffling the feature reduced ROC-AUC.
    print("Calculating validation permutation importance...")
    perm = permutation_importance(
        model,
        X_val[features],
        y_val,
        scoring="roc_auc",
        n_repeats=REPEATS,
        random_state=SEED,
        n_jobs=1,
    )

    perm_df = pd.DataFrame({
        "feature": features,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)

    # 2. Built-in impurity importance, aggregated to original feature names.
    fitted_preprocess = model.named_steps["preprocess"]
    fitted_rf = model.named_steps["classifier"]
    transformed_names = fitted_preprocess.get_feature_names_out()
    raw_importances = fitted_rf.feature_importances_

    if len(transformed_names) != len(raw_importances):
        raise RuntimeError("Transformed feature names do not match RF importances")

    importance_map = {feature: 0.0 for feature in features}

    for transformed_name, value in zip(transformed_names, raw_importances):
        name = str(transformed_name)

        if name.startswith("numeric__"):
            original = name[len("numeric__"):]
        elif name.startswith("categorical__"):
            encoded_name = name[len("categorical__"):]
            matches = [
                col for col in categorical
                if encoded_name == col or encoded_name.startswith(col + "_")
            ]
            if not matches:
                print("Warning: could not map encoded feature:", name)
                continue
            original = max(matches, key=len)
        else:
            continue

        importance_map[original] = importance_map.get(original, 0.0) + float(value)

    impurity_df = pd.DataFrame([
        {"feature": feature, "impurity_importance": value}
        for feature, value in importance_map.items()
    ]).sort_values("impurity_importance", ascending=False)

    combined = perm_df.merge(impurity_df, on="feature", how="outer")
    combined["permutation_rank"] = (
        combined["permutation_importance_mean"]
        .rank(ascending=False, method="min")
    )
    combined["impurity_rank"] = (
        combined["impurity_importance"]
        .rank(ascending=False, method="min")
    )
    combined = combined.sort_values(
        "permutation_importance_mean", ascending=False
    )

    prefix = held_out.lower()
    combined.to_csv(OUT_DIR / f"{prefix}_feature_importance.csv", index=False)

    summary = {
        "run_id": RUN_ID,
        "held_out_category": held_out,
        "dataset": str(TRAIN_FILE),
        "seed": SEED,
        "sample_rows": int(len(sample)),
        "fit_rows": int(len(X_fit)),
        "calibration_rows": int(len(X_cal)),
        "validation_rows": int(len(X_val)),
        "held_out_category_excluded_from_all_partitions": True,
        "validation_roc_auc": float(val_auc),
        "permutation_repeats": REPEATS,
        "permutation_scoring": "roc_auc",
        "top_10_permutation_features": perm_df.head(10).to_dict(orient="records"),
        "top_10_impurity_features": impurity_df.head(10).to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - start, 2),
        "limitations": [
            "Feature importance describes this fitted model and validation sample.",
            "Correlated features can share or mask permutation importance.",
            "Impurity importance can favor features with more possible split points.",
            "Importance does not establish causality or real-world zero-day performance.",
            "The validation partition is used for importance analysis, not as an untouched final evaluation."
        ],
    }
    all_summaries.append(summary)

    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print("\nTop 10 by permutation importance:")
    print(perm_df.head(10).round(6).to_string(index=False))
    print("\nTop 10 by impurity importance:")
    print(impurity_df.head(10).round(6).to_string(index=False))
    print("Elapsed seconds:", summary["elapsed_seconds"])

with open(OUT_DIR / "feature_importance_summary.json", "w") as f:
    json.dump({
        "experiment": "training_side_feature_importance",
        "run_id": RUN_ID,
        "scikit_learn_version": sklearn.__version__,
        "summaries": all_summaries,
    }, f, indent=2)

print("\n=== COMPLETE ===")
print("Report directory:", OUT_DIR)
print("No test CSV was loaded or evaluated.")
