# XGBoost: Complete Technical Documentation

## 1. What is XGBoost?

**XGBoost** (Extreme Gradient Boosting) is a machine learning algorithm that builds multiple decision trees sequentially, where each new tree focuses on correcting the errors of previous trees.

|      Property     |                       Value                        |
|-------------------|----------------------------------------------------|
| Developer         | Tianqi Chen, University of Washington              |
| Year              | 2014                                               |
| Type              | Ensemble learning (gradient boosting)              |
| Industry Adoption | Used by 80% of Kaggle winners; Google, Amazon, Uber|

## 2. How XGBoost Works

### Step-by-Step Process

|Step|            Action             |                 Example                    |
|----|-------------------------------|--------------------------------------------|
| 1  | Start with initial prediction | Average severity = 2.46                    |
| 2  | Calculate errors (residuals)  | Actual=3, Predicted=2.46 → Error=+0.54     |
| 3  | Build tree to predict errors  | If blocked=1 and Junction=1 → Error +0.4   |
| 4  | Add tree to model             | New prediction = 2.46 + (0.1 × 0.4) = 2.50 |
| 5  | Repeat for 100-200 trees      | Each tree focuses on previous mistakes     |
| 6  | Final prediction              | Sum of all weighted trees                  |

### Visual Flow

|         Stage       |              Description              |
|---------------------|---------------------------------------|
| Initial Prediction  | Average severity = 2.46               |
| Tree 1              | Corrects 40% of errors                |
| Tree 2              | Corrects 30% of remaining errors      |
| Tree 3-99           | Continue sequential improvement       |
| Tree 100            | Final corrections                     |
| Final Prediction    | Severity determined by majority vote  |

## 3. Mathematical Foundation

### Core Equation

ŷᵢ = Σ fₖ(xᵢ) for k = 1 to K

| Symbol |              Meaning             |
|--------|----------------------------------|
|   ŷᵢ   | Predicted severity for accident i|
|   xᵢ   | Features of accident i           |
|   fₖ   | k-th decision tree                |
|   K    | Total number of trees (100-200)  |

### Objective Function

Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)

|       Term      |                 Meaning                   |
|-----------------|-------------------------------------------|
| Σ L(yᵢ, ŷᵢ)     | Loss function (measures prediction error) |
| Σ Ω(fₖ)         | Regularization (penalizes complex trees)  |

### Regularization Parameters

|           Parameter             | Symbol |         Effect         |
|---------------------------------|--------|------------------------|
| Minimum loss reduction to split | γ      | Controls tree growth   |
| L2 regularization               | λ      | Shrinks leaf weights   |
| Number of leaves                | T      | Penalizes complex trees|

---

## 4. Model Comparison Results

### Comparison on 1 Million Sample Rows

|       Model         | Accuracy | Training Time (seconds) | Rank |
|--------------------|-----------|-------------------------|------|
| XGBoost            | 83.81%    |          28             |  1   |
| LightGBM           | 83.38%    |          15             |  2   |
| Decision Tree      | 82.77%    |          9              |  3   |
| Neural Network     | 82.71%    |          866            |  4   |
| Random Forest      | 82.05%    |          36             |  5   |
| Logistic Regression| 80.32%    |          124            |  6   |

### Full Data Performance (6.7 Million Rows)

|     Metric    |           Value           |
|---------------|---------------------------|
| Training rows | 5,357,147                 |
| Test rows     | 1,339,287                 |
| Final Accuracy| 83.72%                    |
| Training time | 509 seconds (8.5 minutes) |

---

## 5. Why XGBoost is Better

### Comparison with Other Models

|        Model       |        Key Limitation          |            XGBoost Advantage             |
|--------------------|--------------------------------|------------------------------------------|
| Decision Tree      | Overfits, unstable             | Uses 100+ trees → stable                 |
| Random Forest      | Cannot correct previous errors | Sequential learning → each tree improves |
| Logistic Regression| Linear only, no interactions   | Handles non-linear patterns              |
| Neural Network     | Needs massive data, slow       | Works with less data, faster              |

### Key Advantages

|     Advantage      |                   Explanation                   |
|--------------------|-------------------------------------------------|
| Mixed data types   | Works with numbers, categories, binary features |
| Missing values     | Automatically learns best direction             |
| Regularization     | Prevents overfitting (γ and λ)                  |
| Parallel processing| Uses all CPU cores (`n_jobs=-1`)                |
| Feature importance | Shows which features matter most                |
| Fast inference     | Milliseconds per prediction                     |

---

## 6. Why XGBoost for Accident Data

### Data Characteristics × XGBoost Capabilities

|            Data Feature            | Why XGBoost Handles It Well           |
|------------------------------------|---------------------------------------|
| Mixed types (numbers + categories) | Native support without transformation |
| 6.7 million rows                   | Scales efficiently                    |
| Missing values                     | Learns optimal direction              |
|Non-linear patterns(rain + junction)| Captures feature interactions         |
| Imbalanced classes                 | Built-in class weighting              |

---

## 7. Decision and Conclusion

|       Factor     |                  Verdict                   |
|------------------|--------------------------------------------|
| Highest accuracy |  83.81% (best among 6 models)              |
| Training speed   |  28 seconds on sample, 8.5 minutes on full |
| Inference speed  |  <1 millisecond per prediction             |
| Industry standard|  Used by Google, Amazon, Uber              |
| Interpretability |  Feature importance available              |

**Final Decision:** XGBoost selected as production model for accident severity prediction API.
