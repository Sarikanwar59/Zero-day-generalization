import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GENERIC_RUN = "20261024T061205Z"
EXPLOITS_RUN = "20261024T061555Z"

INPUTS = {
    "Generic": Path(
        f"reports/generic_target_fpr/{GENERIC_RUN}/target_fpr_summary.csv"
    ),
    "Exploits": Path(
        f"reports/exploits_target_fpr/{EXPLOITS_RUN}/target_fpr_summary.csv"
    ),
}

for category, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{category} summary not found: {path}\n"
            "Check the run ID and report directory."
        )

frames = []
for category, path in INPUTS.items():
    df = pd.read_csv(path)
    df["heldout_category"] = category
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

# Source summaries call evaluation false-positive rate "fpr_*".
# Map it to explicit evaluation-FPR names for the combined report.
for source_col, report_col in [
    ("fpr_mean", "evaluation_fpr_mean"),
    ("fpr_sd", "evaluation_fpr_sd"),
    ("fpr_min", "evaluation_fpr_min"),
    ("fpr_max", "evaluation_fpr_max"),
]:
    if source_col in combined.columns:
        if report_col in combined.columns:
            combined[report_col] = combined[report_col].fillna(
                combined[source_col]
            )
        else:
            combined[report_col] = combined[source_col]

required_fpr = ["evaluation_fpr_mean", "evaluation_fpr_sd"]
missing = [c for c in required_fpr if c not in combined.columns]
if missing:
    raise KeyError(f"Missing evaluation FPR columns: {missing}")
if combined["evaluation_fpr_mean"].isna().any():
    raise ValueError("Evaluation FPR mean still contains missing values.")

# Express key operating-point measures in percentage points.
for col in [
    "recall_mean", "recall_sd", "recall_min", "recall_max",
    "evaluation_fpr_mean", "evaluation_fpr_sd",
    "precision_mean", "f1_mean", "accuracy_mean",
]:
    if col in combined.columns:
        combined[col + "_pct"] = combined[col] * 100

# Threshold variability is reported on the score scale, not as a percentage.
combined["recall_range_pp"] = (
    combined["recall_max"] - combined["recall_min"]
) * 100

# Compare ablation with the full model at the same category and target.
comparison_rows = []
for (category, target), group in combined.groupby(
    ["heldout_category", "target_fpr"]
):
    by_variant = group.set_index("variant")
    if not {"full", "no_top5"}.issubset(by_variant.index):
        continue

    full = by_variant.loc["full"]
    ablated = by_variant.loc["no_top5"]

    comparison_rows.append({
        "heldout_category": category,
        "target_fpr": target,
        "full_recall_pct": full["recall_mean_pct"],
        "no_top5_recall_pct": ablated["recall_mean_pct"],
        "recall_delta_no_top5_minus_full_pp": (
            ablated["recall_mean"] - full["recall_mean"]
        ) * 100,
        "full_eval_fpr_pct": full["evaluation_fpr_mean"] * 100,
        "no_top5_eval_fpr_pct": ablated["evaluation_fpr_mean"] * 100,
        "full_recall_sd_pp": full["recall_sd_pct"],
        "no_top5_recall_sd_pp": ablated["recall_sd_pct"],
        "full_recall_range_pp": full["recall_range_pp"],
        "no_top5_recall_range_pp": ablated["recall_range_pp"],
        "full_threshold_sd": full["threshold_sd"],
        "no_top5_threshold_sd": ablated["threshold_sd"],
        "full_roc_auc": full["roc_auc"],
        "no_top5_roc_auc": ablated["roc_auc"],
        "full_average_precision": full["average_precision"],
        "no_top5_average_precision": ablated["average_precision"],
    })

comparison = pd.DataFrame(comparison_rows)

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = Path(f"reports/cross_category_summary/{run_id}")
out_dir.mkdir(parents=True, exist_ok=True)

combined_path = out_dir / "cross_category_all_metrics.csv"
comparison_path = out_dir / "cross_category_operating_points.csv"
metadata_path = out_dir / "metadata.json"

combined.to_csv(combined_path, index=False)
comparison.to_csv(comparison_path, index=False)

metadata = {
    "run_id": run_id,
    "source_reports": {k: str(v) for k, v in INPUTS.items()},
    "categories": ["Generic", "Exploits"],
    "target_fprs": [0.01, 0.02, 0.05, 0.10],
    "purpose": (
        "Descriptive comparison of held-out-category operating points "
        "and calibration sensitivity."
    ),
    "caveats": [
        "The no_top5 feature sets differ between Generic and Exploits.",
        "Calibration replicates overlap and are not independent.",
        "The evaluation cohort is balanced by design.",
        "Precision and F1 are not deployment-prevalence estimates.",
        "Results are exploratory and specific to UNSW-NB15.",
        "This analysis does not establish real-world zero-day detection."
    ]
}
metadata_path.write_text(json.dumps(metadata, indent=2))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

display_cols = [
    "heldout_category", "target_fpr",
    "full_recall_pct", "no_top5_recall_pct",
    "recall_delta_no_top5_minus_full_pp",
    "full_eval_fpr_pct", "no_top5_eval_fpr_pct",
    "full_recall_sd_pp", "no_top5_recall_sd_pp",
    "full_recall_range_pp", "no_top5_recall_range_pp",
    "full_roc_auc", "no_top5_roc_auc",
    "full_average_precision", "no_top5_average_precision",
]

print("\\n=== CROSS-CATEGORY OPERATING-POINT COMPARISON ===")
print(comparison[display_cols].round(4).to_string(index=False))
print(f"\\nAll metrics: {combined_path}")
print(f"Operating-point comparison: {comparison_path}")
print(f"Metadata: {metadata_path}")
