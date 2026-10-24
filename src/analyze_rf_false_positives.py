import json
import os
import glob
import joblib
import numpy as np
import pandas as pd

run_id = "20261024T051225Z"
model_dir = f"models/rf_calibrated/{run_id}"
report_dir = f"reports/rf_calibrated/{run_id}"
data_path = "data/raw/unsw_nb15/UNSW_NB15_testing-set.csv"
results_path = f"{report_dir}/rf_calibrated_heldout_results.json"

with open(results_path) as f:
    saved_data = json.load(f)

saved_results = saved_data["results"]

df = pd.read_csv(data_path, low_memory=False)
df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)

excluded = {"id", "attack_cat", "label"}
feature_cols = [c for c in df.columns if c not in excluded]

numeric_cols = df[feature_cols].select_dtypes(
    include=[np.number]
).columns.tolist()

os.makedirs(report_dir, exist_ok=True)

for category in ["Generic", "Exploits"]:
    result = next(
        (r for r in saved_results
         if r.get("held_out_category") == category),
        None
    )
    if result is None:
        raise ValueError(f"No saved result found for {category}")

    # Locate a saved model using its filename.
    category_key = category.lower()
    candidates = glob.glob(
        f"{model_dir}/**/*{category_key}*.joblib",
        recursive=True
    )

    model = None
    model_path = None

    for candidate in candidates:
        try:
            obj = joblib.load(candidate)
            if hasattr(obj, "predict_proba"):
                model = obj
                model_path = candidate
                break
        except Exception:
            continue

    if model is None:
        print(f"\nCould not identify model for {category}.")
        print("Available model files:")
        for path in glob.glob(
            f"{model_dir}/**/*.joblib", recursive=True
        ):
            print(" ", path)
        raise FileNotFoundError(
            f"Could not match a saved model to {category}"
        )

    normal = df[df["attack_cat"] == "Normal"]
    attacks = df[df["attack_cat"] == category]

    if len(normal) < 5000 or len(attacks) < 5000:
        raise ValueError(
            f"Not enough records for {category}: "
            f"normal={len(normal)}, attacks={len(attacks)}"
        )

    normal_sample = normal.sample(n=5000, random_state=42)
    attack_sample = attacks.sample(n=5000, random_state=42)

    evaluation = pd.concat(
        [normal_sample, attack_sample]
    ).sample(frac=1, random_state=42).copy()

    X = evaluation[feature_cols]
    y = (evaluation["attack_cat"] == category).astype(int).to_numpy()

    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(result["selected_threshold"])
    predictions = (probabilities >= threshold).astype(int)

    evaluation["actual_attack"] = y
    evaluation["predicted_attack"] = predictions
    evaluation["attack_probability"] = probabilities

    false_positives = evaluation[
        (evaluation["actual_attack"] == 0) &
        (evaluation["predicted_attack"] == 1)
    ]
    true_negatives = evaluation[
        (evaluation["actual_attack"] == 0) &
        (evaluation["predicted_attack"] == 0)
    ]

    print(f"\n=== {category}: false-positive analysis ===")
    print("Model:", model_path)
    print("Threshold:", round(threshold, 6))
    print("False positives:", len(false_positives))
    print("True negatives:", len(true_negatives))

    rows = []
    for col in numeric_cols:
        fp = pd.to_numeric(
            false_positives[col], errors="coerce"
        ).dropna()
        tn = pd.to_numeric(
            true_negatives[col], errors="coerce"
        ).dropna()

        if fp.empty or tn.empty:
            continue

        rows.append({
            "feature": col,
            "fp_mean": fp.mean(),
            "tn_mean": tn.mean(),
            "fp_median": fp.median(),
            "tn_median": tn.median(),
            "fp_zero_fraction": (fp == 0).mean(),
            "tn_zero_fraction": (tn == 0).mean(),
        })

    comparison = pd.DataFrame(rows)
    comparison["absolute_mean_difference"] = (
        comparison["fp_mean"] - comparison["tn_mean"]
    ).abs()
    comparison = comparison.sort_values(
        "absolute_mean_difference", ascending=False
    )

    print("\nLargest raw mean differences (descriptive only):")
    print(comparison.head(12).round(4).to_string(index=False))

    out = (
        f"{report_dir}/"
        f"{category_key}_fp_feature_comparison.csv"
    )
    comparison.to_csv(out, index=False)
    print("Saved:", out)

print(
    "\nAnalysis complete. This is exploratory analysis of a test "
    "dataset already examined in previous experiments. "
    "Do not use it to tune thresholds or claim unbiased final "
    "performance."
)
