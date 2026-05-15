# Hyperparameter Tuning Report

**Date:** 2026-05-15 16:28:30

## Class Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| 1 | 67,364 | 1.0% |
| 2 | 5,152,888 | 76.9% |
| 3 | 1,299,243 | 19.4% |
| 4 | 176,939 | 2.6% |

## Best Class Weights Found

| Severity | Weight |
|----------|--------|
| 1 | 2.5 |
| 2 | 1.0 |
| 3 | 1.2 |
| 4 | 3.0 |

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| max_depth | 8 |
| learning_rate | 0.1 |
| n_estimators | 200 |
| subsample | 0.8 |
| colsample_bytree | 0.9 |
| min_child_weight | 1 |

## Weight Configuration Test Results

| Config | Weights | Accuracy | Sev1 Recall | Sev4 Recall |
|--------|---------|----------|-------------|-------------|
| Current | 2.5,1.0,1.2,2.0 | 82.78% | 17.2% | 68.1% |
| Boost_Sev1 | 4.0,1.0,1.2,2.0 | 82.70% | 25.9% | 67.9% |
| Boost_Sev4 | 2.5,1.0,1.2,3.0 | 82.65% | 16.7% | 72.4% |
| Balanced | 3.0,0.8,1.0,2.5 | 82.54% | 23.3% | 72.8% |
| High_Boost | 5.0,0.7,1.0,3.5 | 81.53% | 47.7% | 75.8% |
| Neutral | 1.0,1.0,1.0,1.0 | 83.09% | 10.0% | 57.3% |

## Final Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 83.70% |
| Training Time | 551.07 seconds |
| Severity 1 Recall | 33.0% |
| Severity 2 Recall | 90.2% |
| Severity 3 Recall | 61.8% |
| Severity 4 Recall | 73.6% |

## Classification Report
              precision    recall  f1-score   support

  Severity 1       0.55      0.33      0.41     13473
  Severity 2       0.89      0.90      0.90   1030578
  Severity 3       0.64      0.62      0.63    259848
  Severity 4       0.68      0.74      0.71     35388

    accuracy                           0.84   1339287
   macro avg       0.69      0.65      0.66   1339287
weighted avg       0.83      0.84      0.84   1339287


text

## Confusion Matrix
Predicted
Sev1 Sev2 Sev3 Sev4
Actual Sev1:  4447  8201   782    43
Actual Sev2:  2924 929823 85821 12010
Actual Sev3:   598 98348 160626   276
Actual Sev4:    62  7446  1830 26050

text

## Prediction Distribution

| Severity | Actual % | Predicted % |
|----------|----------|-------------|
| 1 | 1.0% | 0.6% |
| 2 | 76.9% | 77.9% |
| 3 | 19.4% | 18.6% |
| 4 | 2.6% | 2.9% |

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Best Parameters:** `outputs/tuning/best_parameters.csv`
- **Report:** `documentation/tuning_report.md`
