# Unseen Attack Category Evaluation

These are preliminary sampled binary classification experiments. They do not prove real zero-day detection.

| held_out_category   |   training_rows |   test_rows |   accuracy |   precision |   attack_recall |   f1_score |   roc_auc |   false_positive_rate |   missed_attacks |   false_positives |
|:--------------------|----------------:|------------:|-----------:|------------:|----------------:|-----------:|----------:|----------------------:|-----------------:|------------------:|
| Generic             |           20000 |       10000 |     0.8442 |      0.7636 |          0.9970 |     0.8649 |    0.9693 |                0.3086 |               15 |              1543 |
| Exploits            |           20000 |       10000 |     0.6639 |      0.8456 |          0.4010 |     0.5440 |    0.7727 |                0.0732 |             2995 |               366 |
