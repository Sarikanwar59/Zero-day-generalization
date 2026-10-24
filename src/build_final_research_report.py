from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CROSS_DIR = Path("reports/cross_category_summary/20261024T062633Z")
UNCERTAINTY_FILE = Path(
    "reports/calibration_uncertainty/20261024T062915Z/"
    "calibration_uncertainty.csv"
)

OPERATING_FILE = CROSS_DIR / "cross_category_operating_points.csv"
ALL_METRICS_FILE = CROSS_DIR / "cross_category_all_metrics.csv"

for path in [OPERATING_FILE, ALL_METRICS_FILE, UNCERTAINTY_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

operating = pd.read_csv(OPERATING_FILE)
metrics = pd.read_csv(ALL_METRICS_FILE)
uncertainty = pd.read_csv(UNCERTAINTY_FILE)

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = Path("reports/final_research_report") / run_id
out.mkdir(parents=True, exist_ok=False)
(out / "figures").mkdir()

# Save clean copies of the principal tables.
operating.to_csv(out / "operating_point_comparison.csv", index=False)
metrics.to_csv(out / "all_metrics.csv", index=False)
uncertainty.to_csv(out / "calibration_uncertainty.csv", index=False)

# Figure 1: Recall by target FPR, category and model variant.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, category in zip(axes, ["Generic", "Exploits"]):
    subset = operating[operating["heldout_category"] == category]

    for variant, label in [
        ("full", "Full model"),
        ("no_top5", "No-top-5 model"),
    ]:
        recall_col = (
            "full_recall_pct" if variant == "full"
            else "no_top5_recall_pct"
        )
        ax.plot(
            subset["target_fpr"] * 100,
            subset[recall_col],
            marker="o",
            label=label,
        )

    ax.set_title(f"Held-out category: {category}")
    ax.set_xlabel("Target FPR (%)")
    ax.set_xticks([1, 2, 5, 10])
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Mean recall (%)")
fig.suptitle("Recall at different target false-positive rates")
fig.tight_layout()
fig.savefig(out / "figures/recall_by_target_fpr.png", dpi=180)
plt.close(fig)

# Figure 2: Actual evaluation FPR versus recall.
fig, ax = plt.subplots(figsize=(7, 6))

for category in ["Generic", "Exploits"]:
    subset = operating[operating["heldout_category"] == category]

    for variant, label in [
        ("full", f"{category} — full"),
        ("no_top5", f"{category} — no-top-5"),
    ]:
        recall_col = (
            "full_recall_pct" if variant == "full"
            else "no_top5_recall_pct"
        )
        fpr_col = (
            "full_eval_fpr_pct" if variant == "full"
            else "no_top5_eval_fpr_pct"
        )
        ax.plot(
            subset[fpr_col],
            subset[recall_col],
            marker="o",
            label=label,
        )

ax.set_xlabel("Observed evaluation FPR (%)")
ax.set_ylabel("Mean recall (%)")
ax.set_title("Recall versus observed evaluation FPR")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "figures/recall_vs_evaluation_fpr.png", dpi=180)
plt.close(fig)

# Figure 3: Recall variability across calibration replicates.
rec = uncertainty[uncertainty["metric"] == "recall"].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, category in zip(axes, ["Generic", "Exploits"]):
    subset = rec[rec["category"] == category]

    for variant, label in [
        ("full", "Full model"),
        ("no_top5", "No-top-5 model"),
    ]:
        s = subset[subset["variant"] == variant].sort_values("target_fpr")
        x = s["target_fpr"].to_numpy() * 100
        y = s["mean_pct"].to_numpy()
        low = s["ci95_low_pct"].to_numpy()
        high = s["ci95_high_pct"].to_numpy()

        # Descriptive t-based intervals, not population-level guarantees.
        ax.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            capsize=3,
            label=label,
        )

    ax.set_title(category)
    ax.set_xlabel("Target FPR (%)")
    ax.set_xticks([1, 2, 5, 10])
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Recall (%) with descriptive 95% interval")
fig.suptitle("Calibration-replicate variation in recall")
fig.tight_layout()
fig.savefig(out / "figures/recall_calibration_uncertainty.png", dpi=180)
plt.close(fig)

# Build a compact operating-point table in Markdown.
display_cols = [
    "heldout_category",
    "target_fpr",
    "full_recall_pct",
    "no_top5_recall_pct",
    "recall_delta_no_top5_minus_full_pp",
    "full_eval_fpr_pct",
    "no_top5_eval_fpr_pct",
]
table = operating[display_cols].copy()
table.columns = [
    "Category", "Target FPR", "Full recall (%)",
    "No-top-5 recall (%)", "Recall difference (pp)",
    "Full observed FPR (%)", "No-top-5 observed FPR (%)",
]
markdown_table = table.round(3).to_markdown(index=False)

report = f"""# Final Research Results Report

## Project

**Generalizable AI for Zero-Day Attack Detection: A Machine Learning
Framework for Identifying Unknown Network Threats Under Distribution Shift**

## Report scope

This report consolidates existing UNSW-NB15 held-out-category experiments,
target-FPR operating-point results, and calibration-replicate uncertainty
summaries. It does not retrain models or introduce new experiments.

## Experimental comparison

{markdown_table}

## Main observations

1. Performance differs substantially by held-out attack category.
2. Removing the selected top five features improves Generic-category recall
   at lower target-FPR settings in these experiments.
3. The feature-removal effect on Exploits is smaller and varies by operating
   point.
4. Observed evaluation FPR may differ from the requested target FPR.
5. Calibration replicates show variation in thresholds, recall, and observed
   FPR.

## Figures

- `figures/recall_by_target_fpr.png`: Recall across target-FPR settings.
- `figures/recall_vs_evaluation_fpr.png`: Recall versus observed evaluation FPR.
- `figures/recall_calibration_uncertainty.png`: Recall with descriptive
  t-based intervals across calibration replicates.

## Methodological limitations

- Results are specific to the current UNSW-NB15 experiments.
- Calibration samples may overlap and are not independent.
- Evaluation cohorts are fixed within each source experiment.
- Evaluation data were balanced by design; precision and F1 therefore should
  not be interpreted as estimates for deployment traffic prevalence.
- The uncertainty intervals describe calibration-replicate variation, not
  population-level generalization.
- Held-out attack-category evaluation is not proof of detection of real-world
  zero-day attacks.
- Independent dataset, temporal, or cross-network validation remains
  necessary before making stronger generalization claims.

## Source files

- `{OPERATING_FILE}`
- `{ALL_METRICS_FILE}`
- `{UNCERTAINTY_FILE}`

## Reproducibility metadata

The input files are preserved in their original experiment directories.
This report was generated at {datetime.now(timezone.utc).isoformat()}.
"""

(out / "research_report.md").write_text(report, encoding="utf-8")

metadata = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_files": [
        str(OPERATING_FILE),
        str(ALL_METRICS_FILE),
        str(UNCERTAINTY_FILE),
    ],
    "output_directory": str(out),
    "operating_point_rows": int(len(operating)),
    "all_metrics_rows": int(len(metrics)),
    "uncertainty_rows": int(len(uncertainty)),
    "limitations": [
        "Exploratory UNSW-NB15 experiments only",
        "Calibration samples may overlap",
        "Evaluation cohorts fixed within source experiments",
        "No independent real-world zero-day validation",
    ],
}

(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nReport generation complete.")
print(f"Output directory: {out}")
print(f"Operating-point rows: {len(operating)}")
print(f"All-metrics rows: {len(metrics)}")
print(f"Uncertainty rows: {len(uncertainty)}")
print("\nGenerated files:")
for path in sorted(out.rglob("*")):
    if path.is_file():
        print(f"  {path}")
