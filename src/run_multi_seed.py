import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "multi_seed"
MODEL_OUTPUT = ROOT / "models" / "multi_seed"
SEEDS = [7, 21, 42]

EXPERIMENTS = [
    ("generic", ROOT / "src" / "evaluate_unseen.py",
     "unseen_generic_metrics.json", "unseen_generic_model.joblib"),
    ("exploits", ROOT / "src" / "evaluate_unseen_exploits.py",
     "unseen_exploits_metrics.json", "unseen_exploits_model.joblib"),
]

OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

for category, original_script, original_metrics, original_model in EXPERIMENTS:
    source = original_script.read_text()

    for seed in SEEDS:
        temp_script = ROOT / "src" / f"_temp_{category}_{seed}.py"
        metrics_name = f"{category}_seed_{seed}.json"
        model_name = f"{category}_seed_{seed}.joblib"

        modified = re.sub(
            r"^SEED\s*=\s*\d+",
            f"SEED = {seed}",
            source,
            count=1,
            flags=re.M,
        )

        # Redirect output files while retaining the original script unchanged.
        modified = modified.replace(
            f'REPORTS / "{original_metrics}"',
            f'Path("{OUTPUT}") / "{metrics_name}"',
        )
        modified = modified.replace(
            f'MODELS / "{original_model}"',
            f'Path("{MODEL_OUTPUT}") / "{model_name}"',
        )

        temp_script.write_text(modified)
        print(f"\n=== {category.upper()} | seed {seed} ===", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            if result.returncode != 0:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:], file=sys.stderr)
                raise RuntimeError(f"Run failed: {category}, seed {seed}")

            print(result.stdout[-800:])

            metrics_path = OUTPUT / metrics_name
            if not metrics_path.exists():
                raise RuntimeError(f"Metrics file not created: {metrics_path}")

            metrics = json.loads(metrics_path.read_text())
            metrics["seed"] = seed
            metrics_path.write_text(json.dumps(metrics, indent=2))

        finally:
            temp_script.unlink(missing_ok=True)

print("\nAll six runs completed.")
print("Metrics:", OUTPUT)
print("Models:", MODEL_OUTPUT)
