# Generalizable AI for Zero-Day Attack Detection

## Overview

This research project investigates machine-learning detection of held-out
network-attack categories under distribution shift using the UNSW-NB15 dataset.

## Research Questions

- How does detection performance change when an attack category is excluded
  from model training?
- How does recall vary at different target false-positive-rate thresholds?
- How does feature ablation affect detection performance?
- How variable are results across repeated calibration experiments?

## Methods

The project includes data preparation, model training, held-out attack-category
evaluation, threshold calibration, feature-ablation experiments, and reporting.

Experiments include the Generic and Exploits attack categories, multiple target
false-positive-rate thresholds, and comparisons between full-feature and
feature-ablation models.

## Repository Structure

- `src/` — experiment, evaluation, analysis, and reporting scripts
- `reports/` — experiment metrics, metadata, figures, and research reports
- `releases/` — packaged research results
- `requirements.txt` — Python dependencies

## Dataset

The experiments use UNSW-NB15. Place the required CSV files under:

    data/raw/unsw_nb15/

Dataset files and trained model artifacts are excluded from Git.

## Environment Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Review individual scripts for their required inputs and experiment settings
before running them.

## Results

The consolidated research report is stored under:

    reports/final_research_report/

The report includes metrics, operating-point comparisons, calibration
uncertainty summaries, and figures.

## Experimental Results

The following table compares recall for held-out attack categories under
different target false-positive-rate (FPR) thresholds. Results compare the
full-feature model with the `no_top5` feature-ablation model.

### Held-out Generic category

| Target FPR | Full-feature recall (%) | no_top5 recall (%) |
|---:|---:|---:|
| 1% | 8.784 | 20.516 |
| 2% | 22.566 | 44.632 |
| 5% | 81.862 | 93.274 |
| 10% | 100.000 | 99.996 |

### Held-out Exploits category

| Target FPR | Full-feature recall (%) | no_top5 recall (%) |
|---:|---:|---:|
| 1% | 74.638 | 77.756 |
| 2% | 84.232 | 88.798 |
| 5% | 95.946 | 95.538 |
| 10% | 99.050 | 98.742 |

### Interpretation

- For the held-out `Generic` category, `no_top5` achieved higher recall at
  target FPR levels of 1%, 2%, and 5%.
- For the held-out `Exploits` category, `no_top5` achieved higher recall at
  target FPR levels of 1% and 2%, but slightly lower recall at 5% and 10%.
- Target FPR is the calibration threshold; the observed evaluation FPR may
  differ from that target.
- These are exploratory experiments using UNSW-NB15. They do not establish
  real-world zero-day detection capability. Independent datasets and
  temporal or cross-network validation are needed.

## Reproducibility and Research Artifacts

### Environment

The experiments were developed in Python using scikit-learn, pandas,
NumPy, Matplotlib, seaborn, and joblib. The development environment used
Python 3.9.25 in a virtual environment on Amazon Linux 2023.

For a new environment, clone the repository and install the declared
dependencies:

```bash
git clone https://github.com/Sarikanwar59/Zero-day-generalization.git
cd Zero-day-generalization

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset requirements

The UNSW-NB15 dataset was used for the held-out attack-category
experiments. Dataset files are not included in this repository. Obtain
the dataset from its official source and place the required CSV files
under `data/raw/unsw_nb15/`.

Expected files used during development include:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Check the filenames and expected columns before running the experiments.
Dataset versions, preprocessing, random seeds, and sampling choices can
affect the resulting metrics.

### Experiment scripts

The repository includes scripts for the UNSW-NB15 training and
held-out-category evaluations:

- `src/train_unsw.py` — trains the UNSW-NB15 baseline pipeline.
- `src/evaluate_unseen.py` — evaluates a held-out `Generic` category.
- `src/evaluate_unseen_exploits.py` — evaluates a held-out `Exploits`
  category.

Inspect each script before execution to confirm its input paths,
sampling configuration, random seeds, and output locations. Run scripts
from the repository root. Re-running experiments may overwrite generated
models or reports, depending on the script configuration.

### Published research artifacts

The final report and analysis outputs from the recorded experiment run
are preserved under:

`reports/final_research_report/20261024T082720Z/`

The report directory contains:

- `research_report.md` — written summary and interpretation.
- `all_metrics.csv` — collected experiment metrics.
- `operating_point_comparison.csv` — operating-point comparisons.
- `calibration_uncertainty.csv` — calibration uncertainty results.
- `metadata.json` — recorded experiment metadata.
- `recall_by_target_fpr.png` — recall at target FPR thresholds.
- `recall_vs_evaluation_fpr.png` — recall versus evaluation FPR.
- `recall_calibration_uncertainty.png` — calibration uncertainty plot.

A clean results archive is available at:

`releases/zero-day-research-results-clean-20261024.tar.gz`

Verify its integrity with:

```bash
sha256sum releases/zero-day-research-results-clean-20261024.tar.gz
```

Expected SHA-256:

`7318ce11c5fa8cd8dc70ab05a07eba219d6e434cff660186d9e30093d669e570`

### Interpretation and reproducibility notes

The README tables report recall for held-out `Generic` and `Exploits`
attack categories at specified target FPR thresholds. A target FPR is a
calibration setting, not a guarantee of the observed evaluation FPR.

The `no_top5` experiment is a feature-ablation comparison; its results
should not be interpreted as evidence that removing features generally
improves detection. Performance depends on the attack category and
operating point.

The published results represent a specific exploratory experiment run.
Exact reproduction requires matching the dataset, preprocessing,
sampling, random seeds, software versions, and evaluation configuration.
Independent dataset, temporal, and cross-network validation remain
necessary before making claims about real-world zero-day detection.

## Limitations

These are exploratory experiments on UNSW-NB15 using held-out attack
categories. They do not establish reliable detection of real-world zero-day
attacks.

Calibration samples may overlap, evaluation cohorts are fixed, and observed
false-positive rates may differ from target thresholds. Independent datasets,
temporal evaluation, and cross-network validation are needed to assess
generalization more robustly.
