# Hyperparameter Tuning Report

**Date:** 2026-05-13 16:19:04

## Summary

|     Metric    |    Value   |
|---------------|------------|
| Test Accuracy | 70.27%     |
| Training rows | 5,357,147  |
| Testing rows  | 1,339,287  |
| Features      |     41     |

## Best Parameters Found

|     Parameter    | Value |
|------------------|-------|
| max_depth        |  N/A  |
| learning_rate    |  N/A  |
| n_estimators     |  N/A  |
| subsample        |  N/A  |
| colsample_bytree |  N/A  |
| min_child_weight |  N/A  |

## Classification Report
              precision    recall  f1-score   support

  Severity 1       0.10      0.93      0.19     13473
  Severity 2       0.96      0.66      0.78   1030578
  Severity 3       0.50      0.82      0.63    259848
  Severity 4       0.38      0.85      0.52     35388

    accuracy                           0.70   1339287
   macro avg       0.49      0.82      0.53   1339287
weighted avg       0.85      0.70      0.74   1339287


text

## Confusion Matrix
[[ 12495    379    560     39]
 [ 91859 684363 207486  46870]
 [ 15470  27328 214200   2850]
 [   821   2249   2265  30053]]

text

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Report:** `documentation/tuning_report.md`
