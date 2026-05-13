# Model Training Report

**Date:** 2026-05-08 17:37:14

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | XGBoost |
| Training rows | 5,357,147 |
| Testing rows | 1,339,287 |
| Features | 41 |
| Train/Test split | 80% / 20% |

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 83.19% |
| Training Time | 221.47 seconds |

## Classification Report

```
              precision    recall  f1-score   support

  Severity 1       0.71      0.12      0.21     13473
  Severity 2       0.86      0.93      0.90   1030578
  Severity 3       0.65      0.51      0.57    259848
  Severity 4       0.96      0.56      0.71     35388

    accuracy                           0.83   1339287
   macro avg       0.80      0.53      0.60   1339287
weighted avg       0.82      0.83      0.82   1339287

```

## Output

- **Model file:** `outputs/models/final_model.pkl`
- **Ready for API deployment**
