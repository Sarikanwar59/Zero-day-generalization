
#!/usr/bin/env python3
"""Reproducible held-out-category experiments with pre-run provenance."""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import sklearn

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "raw" / "unsw_nb15"
TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

SEEDS = [7, 21, 42]
EXPERIMENTS = {
    "Generic": {
        "script": BASE / "src" / "evaluate_unseen.py",
        "model_name": "unseen_generic_model.joblib",
        "metrics_name": "unseen_generic_metrics.json",
    },
    "Exploits": {
        "script": BASE / "src" / "evaluate_unseen_exploits.py",
        "model_name": "unseen_exploits_model.joblib",
        "metrics_name": "unseen_exploits_metrics.json",
    },
}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def require_files():
    required = [TRAIN_FILE, TEST_FILE]
    required += [item["script"] for item in EXPERIMENTS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files missing:\n" + "\n".join(missing))

def build_script(source_path, heldout, seed, model_path, metrics_path, destination):
    source = source_path.read_text()

    # Change only the seed and the final model/metrics output destinations.
    source, seed_count = re.subn(
        r"(?m)^SEED\s*=\s*\d+\s*$",
        f"SEED = {seed}",
        source,
        count=1,
    )
    if seed_count != 1:
        raise RuntimeError(f"Could not safely replace SEED in {source_path}")

    model_name = EXPERIMENTS[heldout]["model_name"]
    metrics_name = EXPERIMENTS[heldout]["metrics_name"]

    if f'"{model_name}"' not in source:
        raise RuntimeError(f"Model output name not found in {source_path}")
    if f'"{metrics_name}"' not in source:
        raise RuntimeError(f"Metrics output name not found in {source_path}")

    # Absolute paths on the right-hand side override the existing Path prefix.
    source = source.replace(f'"{model_name}"', repr(str(model_path)))
    source = source.replace(f'"{metrics_name}"', repr(str(metrics_path)))

    destination.write_text(source)
    return destination

def main():
    require_files()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = BASE / "reports" / "reproducible_runs" / run_id
    model_root = BASE / "models" / "reproducible_runs" / run_id
    script_root = run_root / "generated_scripts"

    run_root.mkdir(parents=True, exist_ok=False)
    model_root.mkdir(parents=True, exist_ok=False)
    script_root.mkdir(parents=True, exist_ok=False)

    config = {
        "seeds": SEEDS,
        "held_out_categories": list(EXPERIMENTS.keys()),
        "train_sample_size": 20000,
        "test_per_class": 5000,
        "expected_test_rows": 10000,
        "dataset_paths": {
            "training": str(TRAIN_FILE.relative_to(BASE)),
            "testing": str(TEST_FILE.relative_to(BASE)),
        },
    }

    # Hashes are captured before any experiment starts.
    pre_run_manifest = {
        "run_id": run_id,
        "captured_at_utc": utc_now(),
        "dataset_sha256": {
            "training": sha256(TRAIN_FILE),
            "testing": sha256(TEST_FILE),
        },
        "evaluation_script_sha256": {
            category: sha256(item["script"])
            for category, item in EXPERIMENTS.items()
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    manifest_path = run_root / "pre_run_manifest.json"
    manifest_path.write_text(json.dumps(pre_run_manifest, indent=2) + "\n")

    print(f"Run ID: {run_id}")
    print(f"Manifest written before experiments: {manifest_path}")
    print("Dataset SHA-256:")
    print(json.dumps(pre_run_manifest["dataset_sha256"], indent=2))

    run_records = []

    for category, item in EXPERIMENTS.items():
        for seed in SEEDS:
            tag = f"{category.lower()}_seed_{seed}"
            model_path = model_root / f"{tag}.joblib"
            metrics_path = run_root / f"{tag}.json"
            log_path = run_root / f"{tag}.log"
            generated_script = script_root / f"{tag}.py"

            build_script(
                item["script"],
                category,
                seed,
                model_path,
                metrics_path,
                generated_script,
            )

            script_hash = sha256(generated_script)
            started_at = utc_now()

            print(f"\n=== Starting {tag} ===")
            with log_path.open("w") as log:
                process = subprocess.run(
                    [sys.executable, str(generated_script)],
                    cwd=BASE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=os.environ.copy(),
                )

            finished_at = utc_now()
            record = {
                "run_id": run_id,
                "experiment": tag,
                "held_out_category": category,
                "seed": seed,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "exit_code": process.returncode,
                "generated_script_sha256": script_hash,
                "log_path": str(log_path.relative_to(BASE)),
                "model_path": str(model_path.relative_to(BASE)),
                "metrics_path": str(metrics_path.relative_to(BASE)),
                "pre_run_manifest_path": str(manifest_path.relative_to(BASE)),
            }

            if process.returncode != 0:
                record["status"] = "failed"
                run_records.append(record)
                print(f"FAILED: {tag}; inspect {log_path}")
                continue

            if not metrics_path.is_file():
                record["status"] = "failed"
                record["error"] = "Evaluation finished without creating metrics JSON."
                run_records.append(record)
                print(f"FAILED: {tag}; metrics file missing")
                continue

            with metrics_path.open() as handle:
                metrics = json.load(handle)

            provenance = {
                "run": record,
                "pre_run_manifest": str(manifest_path.relative_to(BASE)),
                "dataset_sha256": pre_run_manifest["dataset_sha256"],
                "evaluation_script_sha256": pre_run_manifest[
                    "evaluation_script_sha256"
                ],
                "runner_sha256": pre_run_manifest["runner_sha256"],
                "config_sha256": pre_run_manifest["config_sha256"],
                "generated_script_sha256": script_hash,
                "note": (
                    "Dataset and source hashes were captured before experiments. "
                    "This documents file identity, not official dataset authenticity "
                    "or real-world zero-day detection validity."
                ),
            }
            metrics["provenance"] = provenance
            metrics["reproducible_run_id"] = run_id

            with metrics_path.open("w") as handle:
                json.dump(metrics, handle, indent=2)
                handle.write("\n")

            record["status"] = "success"
            run_records.append(record)
            print(f"SUCCESS: {tag}")

    summary = {
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "expected_runs": len(EXPERIMENTS) * len(SEEDS),
        "successful_runs": sum(r["status"] == "success" for r in run_records),
        "failed_runs": sum(r["status"] == "failed" for r in run_records),
        "manifest": str(manifest_path.relative_to(BASE)),
        "runs": run_records,
    }

    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== RUN SUMMARY ===")
    print(json.dumps({
        "run_id": run_id,
        "successful_runs": summary["successful_runs"],
        "failed_runs": summary["failed_runs"],
        "summary_file": str(summary_path.relative_to(BASE)),
    }, indent=2))

    if summary["failed_runs"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
