# Final Research Results Report

## Project

**Generalizable AI for Zero-Day Attack Detection: A Machine Learning
Framework for Identifying Unknown Network Threats Under Distribution Shift**

## Report scope

This report consolidates existing UNSW-NB15 held-out-category experiments,
target-FPR operating-point results, and calibration-replicate uncertainty
summaries. It does not retrain models or introduce new experiments.

## Experimental comparison

| Category   |   Target FPR |   Full recall (%) |   No-top-5 recall (%) |   Recall difference (pp) |   Full observed FPR (%) |   No-top-5 observed FPR (%) |
|:-----------|-------------:|------------------:|----------------------:|-------------------------:|------------------------:|----------------------------:|
| Exploits   |         0.01 |            74.638 |                77.756 |                    3.118 |                   0.878 |                       0.856 |
| Exploits   |         0.02 |            84.232 |                88.798 |                    4.566 |                   1.64  |                       2.024 |
| Exploits   |         0.05 |            95.946 |                95.538 |                   -0.408 |                   4.408 |                       4.494 |
| Exploits   |         0.1  |            99.05  |                98.742 |                   -0.308 |                   9.258 |                       9.24  |
| Generic    |         0.01 |             8.784 |                20.516 |                   11.732 |                   0.988 |                       0.896 |
| Generic    |         0.02 |            22.566 |                44.632 |                   22.066 |                   2.092 |                       1.966 |
| Generic    |         0.05 |            81.862 |                93.274 |                   11.412 |                   4.97  |                       4.894 |
| Generic    |         0.1  |           100     |                99.996 |                   -0.004 |                  10.446 |                      10.304 |

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

- `reports/cross_category_summary/20261024T062633Z/cross_category_operating_points.csv`
- `reports/cross_category_summary/20261024T062633Z/cross_category_all_metrics.csv`
- `reports/calibration_uncertainty/20261024T062915Z/calibration_uncertainty.csv`

## Reproducibility metadata

The input files are preserved in their original experiment directories.
This report was generated at 2026-10-24T08:27:21.030791+00:00.
