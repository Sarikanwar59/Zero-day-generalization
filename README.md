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

## Limitations

These are exploratory experiments on UNSW-NB15 using held-out attack
categories. They do not establish reliable detection of real-world zero-day
attacks.

Calibration samples may overlap, evaluation cohorts are fixed, and observed
false-positive rates may differ from target thresholds. Independent datasets,
temporal evaluation, and cross-network validation are needed to assess
generalization more robustly.
