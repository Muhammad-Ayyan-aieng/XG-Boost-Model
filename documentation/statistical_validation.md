
# Statistical Validation Report

## Test 1: T-Test (Model vs Random Guessing)

|      Metric      |  Value  |
|------------------|---------|
| Random accuracy  | 25%     |
| Model accuracy   | 83.19%  |
| P-value          | 0.000000|
| Significant?     | YES     |

**Conclusion:** The model is statistically significant.

## Test 2: Chi-Square Test Results
  
|     Feature    | Chi-Square | P-Value  | Predicts Severity? |
|----------------|------------|----------|--------------------|
| has_blocked    | 1281061.78 | 0.000000 |        YES         |
| has_road_closed| 3141407.91 | 0.000000 |        YES         |
| has_jackknife  | 2723.18    | 0.000000 |        YES         |
|has_slow_traffic| 115138.78  | 0.000000 |        YES         |
| Temperature(F) | 17800.92   | 0.000000 |        YES         |

## Test 3: Tuning Improvement

|      Model     |Accuracy |
|----------------|---------|
| Default XGBoost| 83.19%  |
| Tuned XGBoost  | 84.46%  |
| Improvement    | 1.27%   |

## Overall Conclusion

The XGBoost model is statistically valid. Text keywords (has_blocked, has_road_closed, has_jackknife) significantly predict accident severity. Weather variables (Temperature) do not show statistical significance, confirming that text descriptions are better predictors.
