# Threshold Analysis Interpretation

## Experiment status
Exploratory threshold analysis on fixed withheld-category test samples.

## Generic
At threshold 0.70:
- Attack recall: 94.60%
- False-positive rate: 9.58%
- Missed attacks: 270 of 5,000
- False positives: 479 of 5,000 normal records

## Exploits
At threshold 0.70:
- Attack recall: 25.40%
- False-positive rate: 9.20%
- Missed attacks: 3,730 of 5,000
- False positives: 460 of 5,000 normal records

## Limitations
- Thresholds were examined on the evaluation samples.
- These results must not be presented as independent threshold validation.
- The Exploits model shows weak discrimination (ROC-AUC approximately 0.637).
- Withheld attack categories are a proxy for generalization, not proof of real zero-day detection.
- Dataset provenance remains unverified against an authoritative checksum.

## Next experiment
Select the threshold using a separate validation set containing only training-known categories.
Freeze the threshold before evaluating on the untouched withheld-category test set.
