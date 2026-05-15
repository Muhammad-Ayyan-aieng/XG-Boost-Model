# Model Training Report

**Date:** 2026-05-15 12:27:45

## Class Distribution

| Severity |   Count   | Percentage |
|----------|-----------|------------|
|     1    | 67,364    | 1.0%       |
|     2    | 5,152,888 | 76.9%      |
|     3    | 1,299,243 | 19.4%      |
|     4    | 176,939   | 2.6%       |

## Class Weights Used

| Severity | Weight |
|----------|--------|
|    1     | 2.50   |
|    2 |     1.00   |
|    3     | 1.20   |
|    4     | 2.00   |

## Training Configuration

|   Parameter   |    Value   |
|---------------|------------|
| Model         | XGBoost    |
| Training rows | 5,357,147  |
| Testing rows  | 1,339,287  |
| Features      | 41         |
| n_estimators  | 200        |
| max_depth     | 6          |
| learning_rate | 0.07       |

## Performance

|     Metric    |      Value      |
|---------------|-----------------|
| Test Accuracy | 83.13%          |
| Training Time | 461.57 seconds  |

## Prediction Distribution

| Severity | Actual % | Predicted % |
|----------|----------|-------------|
|     1    | 1.0%     | 0.4%        |
|     2    | 76.9%    | 79.6%       |
|     3    | 19.4%    | 17.7%       |
|     4    | 2.6%     | 2.4%        |

## Classification Report

```
              precision    recall  f1-score   support

  Severity 1       0.54      0.21      0.30     13473
  Severity 2       0.88      0.91      0.89   1030578
  Severity 3       0.63      0.58      0.60    259848
  Severity 4       0.75      0.68      0.71     35388

    accuracy                           0.83   1339287
   macro avg       0.70      0.59      0.63   1339287
weighted avg       0.82      0.83      0.83   1339287

```

## Confusion Matrix

```
                 Predicted
              Sev1  Sev2  Sev3  Sev4
Actual Sev1:  2831  9647   993     2
Actual Sev2:  1889 936782 83931  7976
Actual Sev3:   505 109509 149785    49
Actual Sev4:    47  9579  1796 23966
```

## Output Files

- `outputs/models/final_model.pkl`
- `outputs/xgboost_model.pkl`
