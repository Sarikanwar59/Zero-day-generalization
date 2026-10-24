from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t

SOURCES = {
    "Generic": Path(
        "reports/generic_target_fpr/20261024T061205Z/target_fpr_results.csv"
    ),
    "Exploits": Path(
        "reports/exploits_target_fpr/20261024T061555Z/target_fpr_results.csv"
    ),
}

OUT = Path("reports/calibration_uncertainty")
OUT.mkdir(parents=True, exist_ok=True)

def mean_interval(values):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    n = len(x)
    mean = float(np.mean(x)) if n else np.nan
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    if n > 1:
        half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = np.nan
    return {
        "n_replicates": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": low,
        "ci95_high": high,
        "min": float(np.min(x)) if n else np.nan,
        "max": float(np.max(x)) if n else np.nan,
    }

rows = []

for category, path in SOURCES.items():
    if not path.exists():
        print(f"SKIP missing source: {path}")
        continue

    df = pd.read_csv(path)
    required = {"variant", "target_fpr", "recall"}
    if not required.issubset(df.columns):
        print(f"\n{category}: columns found: {df.columns.tolist()}")
        raise SystemExit(
            f"{path} lacks expected columns {sorted(required)}. "
            "Inspect the source CSV before proceeding."
        )

    fpr_col = next(
        (c for c in ["evaluation_fpr", "fpr"] if c in df.columns),
        None,
    )
    if fpr_col is None:
        print(f"\n{category}: columns found: {df.columns.tolist()}")
        raise SystemExit(
            f"No evaluation FPR column found in {path}."
        )

    for (variant, target), group in df.groupby(["variant", "target_fpr"]):
        for metric, col in [("recall", "recall"), ("evaluation_fpr", fpr_col)]:
            stats = mean_interval(group[col])
            rows.append({
                "category": category,
                "variant": variant,
                "target_fpr": float(target),
                "metric": metric,
                **stats,
            })

result = pd.DataFrame(rows)
if result.empty:
    raise SystemExit(
        "No results processed. Check source paths and column names."
    )

result["mean_pct"] = result["mean"] * 100
result["sd_percentage_points"] = result["sd"] * 100
result["ci95_low_pct"] = result["ci95_low"] * 100
result["ci95_high_pct"] = result["ci95_high"] * 100
result["min_pct"] = result["min"] * 100
result["max_pct"] = result["max"] * 100

run_dir = OUT / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
run_dir.mkdir(parents=True, exist_ok=True)
csv_path = run_dir / "calibration_uncertainty.csv"
result.to_csv(csv_path, index=False)

metadata = {
    "sources": {k: str(v) for k, v in SOURCES.items()},
    "method": "Mean and t-based interval across calibration replicates",
    "caveats": [
        "Calibration samples may overlap and are not independent.",
        "Evaluation cohorts are fixed within each source experiment.",
        "Intervals describe calibration-replicate variation, not population-level generalization.",
        "Results are exploratory and specific to the current UNSW-NB15 experiments.",
    ],
}
(run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

display_cols = [
    "category", "variant", "target_fpr", "metric", "n_replicates",
    "mean_pct", "sd_percentage_points", "ci95_low_pct", "ci95_high_pct",
    "min_pct", "max_pct",
]
print(result[display_cols].round(4).to_string(index=False))
print(f"\nSaved: {csv_path}")
print(f"Metadata: {run_dir / 'metadata.json'}")
