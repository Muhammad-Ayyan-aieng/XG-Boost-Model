# Statistical Validation Report (Final Model)

**Date:** 2026-05-15 16:37:10

## Model Overview

| Metric | Value |
|--------|-------|
| Model | XGBoost with Balanced Class Weights |
| Training rows | 5,357,147 |
| Testing rows | 1,339,287 |
| Features | 41 |

## Class Distribution in Training Data

| Severity | Percentage | Weight Used |
|----------|------------|-------------|
| 1 | 1.0% | 2.5 |
| 2 | 76.9% | 1.0 |
| 3 | 19.4% | 1.2 |
| 4 | 2.6% | 2.0 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 83.13% |
| Severity 1 Recall | 21.0% |
| Severity 2 Recall | 90.9% |
| Severity 3 Recall | 57.6% |
| Severity 4 Recall | 67.7% |
| Severity 4 Precision | 74.9% |
| P-value (vs random) | 0.000000 |

## Test 1: T-Test (Model vs Random Guessing)

- Random guessing accuracy: 25%
- Model accuracy: 83.13%
- P-value: 0.000000

**Conclusion:** The model is statistically significant (p < 0.05). The probability of achieving 83.13% accuracy by random chance is less than 0.001%.

## Test 2: Chi-Square Test Results

| Feature | Chi-Square | P-Value | Predicts Severity? |
|---------|------------|---------|-------------------|
| has_blocked | 1281061.78 | 0.000000 | YES |
| has_road_closed | 3141407.91 | 0.000000 | YES |
| has_jackknife | 2723.18 | 0.000000 | YES |
| Temperature(F) | 17800.92 | 0.000000 | YES |

## Prediction Distribution

| Severity | Actual % | Predicted % | Difference |
|----------|----------|-------------|------------|
| 1 | 1.0% | 0.4% | -0.6% |
| 2 | 76.9% | 79.6% | +2.6% |
| 3 | 19.4% | 17.7% | -1.7% |
| 4 | 2.6% | 2.4% | -0.3% |

## Classification Report
              precision    recall  f1-score   support

  Severity 1       0.54      0.21      0.30     13473
  Severity 2       0.88      0.91      0.89   1030578
  Severity 3       0.63      0.58      0.60    259848
  Severity 4       0.75      0.68      0.71     35388

    accuracy                           0.83   1339287
   macro avg       0.70      0.59      0.63   1339287
weighted avg       0.82      0.83      0.83   1339287


text

## Confusion Matrix
Predicted
Sev1 Sev2 Sev3 Sev4
Actual Sev1:  2831  9647   993     2
Actual Sev2:  1889 936782 83931  7976
Actual Sev3:   505 109509 149785    49
Actual Sev4:    47  9579  1796 23966

text

## Conclusion

The XGBoost model with balanced class weights is statistically valid. Key findings:

- The model achieves 67.7% recall on severe accidents (Level 4)
- The model achieves 57.6% recall on serious accidents (Level 3)
- Text keywords ('has_blocked', 'has_road_closed', 'has_jackknife') are statistically significant predictors (p < 0.001)
- Temperature shows statistical significance but with weaker predictive power

## Output Files

- **Report:** `documentation/statistical_validation.md`
- **Model:** `outputs/models/final_model.pkl`
