# Hyperparameter Tuning Report

**Date:** 2026-05-11 17:27:49

## Configuration

|       Parameter       |        Value         |
|-----------------------|----------------------|
| Training rows         | 5,357,147            |
| Testing rows          | 1,339,287            |
| Tuning sample         | 500,000              |
| Cross-validation folds| 5                    |
| Combinations tested   | 20                   |

## Best Parameters Found

|    Parameter    | Value |
|-----------------|-------|
| max_depth       | 10    |
| learning_rate   | 0.1   |
| n_estimators    | 150   |
| subsample       | 0.8   |
| colsample_bytree| 0.7   |
| min_child_weight| 3     |

## Results

|      Model    |      Accuracy       |
|---------------|---------------------|
| Tuned XGBoost | 84.46%              |

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Report:** `documentation/tuning_report.md`

## Next Step

Run `compare_default_tuned.py` to compare with default model.
