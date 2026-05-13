# Statistical Validation Report (Final Model)

**Date:** 2026-05-13 16:35:30

## Model Overview

| Metric | Value |
|--------|-------|
| Model | XGBoost with Class Weights |
| Training rows | 5,357,147 |
| Testing rows | 1,339,287 |
| Features | 41 |

## Class Distribution in Training Data

| Severity | Percentage | Weight Used |
|----------|------------|-------------|
| 1 | 1.0% | 24.85 |
| 2 | 76.9% | 0.32 |
| 3 | 19.4% | 1.29 |
| 4 | 2.6% | 9.46 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 70.19% |
| Severity 1 Recall | 92.7% |
| Severity 2 Recall | 66.3% |
| Severity 3 Recall | 82.5% |
| Severity 4 Recall | 85.0% |
| Severity 4 Precision | 38.0% |
| P-value (vs random) | 0.000000 |

## Test 1: T-Test (Model vs Random Guessing)

- Random guessing accuracy: 25%
- Model accuracy: 70.19%
- P-value: 0.000000

**Conclusion:** The model is statistically significant (p < 0.05).

## Test 2: Chi-Square Test Results

| Feature | Chi-Square | P-Value | Predicts Severity? |
|---------|------------|---------|-------------------|
| has_blocked | 1281061.78 | 0.000000 | YES |
| has_road_closed | 3141407.91 | 0.000000 | YES |
| has_jackknife | 2723.18 | 0.000000 | YES |
| Temperature(F) | 17800.92 | 0.000000 | YES |

## Classification Report
              precision    recall  f1-score   support

  Severity 1       0.10      0.93      0.18     13473
  Severity 2       0.96      0.66      0.78   1030578
  Severity 3       0.50      0.83      0.63    259848
  Severity 4       0.38      0.85      0.53     35388

    accuracy                           0.70   1339287
   macro avg       0.49      0.82      0.53   1339287
weighted avg       0.85      0.70      0.74   1339287


text

## Confusion Matrix
Predicted
Sev1 Sev2 Sev3 Sev4
Actual Sev1: 12484   379   576    34
Actual Sev2: 93119 683004 207988 46467
Actual Sev3: 16008 26898 214437  2505
Actual Sev4:   847  2208  2267 30066

text

## Conclusion

The XGBoost model with class weights is statistically valid. It achieves 85.0% recall on severe accidents (Level 4), which is the primary objective of this system.

The model successfully detects:
- 85.0% of severe accidents
- 82.5% of serious accidents
- 92.7% of minor accidents

## Output Files

- **Report:** `documentation/statistical_validation.md`
