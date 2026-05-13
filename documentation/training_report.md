# Model Training Report

**Date:** 2026-05-13 13:59:17

## Class Distribution

| Severity |   Count   | Percentage|
|----------|-----------|-----------|
|    1     | 67,364    | 1.0%      |
|    2     | 5,152,888 | 76.9%     |
|    3     | 1,299,243 | 19.4%     |
|    4     | 176,939   | 2.6%      |

## Class Weights Used

| Severity| Weight  |
|---------|---------|
|    1    | 24.8518 |
|    2    | 0.3249  |
|    3    | 1.2885  |
|    4    | 9.4615  |

## Training Configuration

|   Parameter   |   Value   |
|---------------|-----------|
| Model         | XGBoost   |
| Training rows | 5,357,147 |
| Testing rows  | 1,339,287 |
| Features      | 41        |
| n_estimators  | 300       |
| max_depth     | 8         |
| learning_rate | 0.05      |

## Performance

|     Metric    |      Value     |
|---------------|----------------|
| Test Accuracy | 70.19%         |
| Training Time | 926.64 seconds |

## Classification Report

```
              precision    recall  f1-score   support

  Severity 1       0.10      0.93      0.18     13473
  Severity 2       0.96      0.66      0.78   1030578
  Severity 3       0.50      0.83      0.63    259848
  Severity 4       0.38      0.85      0.53     35388

    accuracy                           0.70   1339287
   macro avg       0.49      0.82      0.53   1339287
weighted avg       0.85      0.70      0.74   1339287

```

## Confusion Matrix

```
                 Predicted
              Sev1  Sev2  Sev3  Sev4
Actual Sev1: 12484   379   576    34
Actual Sev2: 93119 683004 207988 46467
Actual Sev3: 16008 26898 214437  2505
Actual Sev4:   847  2208  2267 30066
```

## Output Files

- `outputs/models/final_model.pkl`
- `outputs/xgboost_model.pkl`
