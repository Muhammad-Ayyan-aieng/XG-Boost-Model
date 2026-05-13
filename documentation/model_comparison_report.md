# Model Comparison Report

**Date:** 2026-05-08 17:05:30

**Sample Size:** 1,000,000 rows

**Train/Test Split:** 80% / 20%

## Results

|       Model        | Accuracy | Train Time (s)| Inference Time (ms) |
|--------------------|----------|---------------|---------------------|
| xgboost            | 83.76%   | 29.35         | 948.84              |
| lightgbm           | 83.43%   | 16.59         | 1903.34             |
| decision_tree      | 82.77%   | 13.61         | 412.08              |
| neural_network     | 82.53%   | 548.42        | 1771.83             |
| random_forest      | 82.11%   | 44.67         | 688.83              |
| logistic_regression| 80.23%   | 127.65        | 460.83              |

## Best Model

**xgboost** achieved the highest accuracy with **83.76%**.
