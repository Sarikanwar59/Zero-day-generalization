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

## Limitations

These are exploratory experiments on UNSW-NB15 using held-out attack
categories. They do not establish reliable detection of real-world zero-day
attacks.

Calibration samples may overlap, evaluation cohorts are fixed, and observed
false-positive rates may differ from target thresholds. Independent datasets,
temporal evaluation, and cross-network validation are needed to assess
generalization more robustly.
