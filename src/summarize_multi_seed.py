import json
import pandas as pd
from pathlib import Path

folder = Path("reports/multi_seed")
rows = []

for path in sorted(folder.glob("*_seed_*.json")):
    m = json.loads(path.read_text())
    tn, fp = m["confusion_matrix"][0]
    fn, tp = m["confusion_matrix"][1]

    rows.append({
        "category": m["held_out_category"],
        "seed": m["seed"],
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "attack_recall": m["recall"],
        "f1_score": m["f1_score"],
        "roc_auc": m["roc_auc"],
        "false_positive_rate": fp / (fp + tn),
        "missed_attacks": fn,
    })

df = pd.DataFrame(rows)
df.to_csv(folder / "all_runs.csv", index=False)

metrics = [
    "accuracy",
    "precision",
    "attack_recall",
    "f1_score",
    "roc_auc",
    "false_positive_rate",
    "missed_attacks",
]

summary = df.groupby("category")[metrics].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()

summary.to_csv(folder / "summary_statistics.csv", index=False)

with open(folder / "summary_statistics.md", "w") as f:
    f.write("# Multi-Seed Withheld-Category Evaluation\n\n")
    f.write("Three random seeds per attack category.\n\n")
    f.write("Standard deviation is the sample standard deviation.\n\n")
    f.write(summary.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n\nThese preliminary experiments do not establish real zero-day detection.\n")

print("\n=== INDIVIDUAL RUNS ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== MEAN AND STANDARD DEVIATION ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved reports:")
print(folder / "all_runs.csv")
print(folder / "summary_statistics.csv")
print(folder / "summary_statistics.md")
