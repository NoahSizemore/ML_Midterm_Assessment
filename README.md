# Midterm Project: Predicting Student Performance

## Author Comments:

* This is a repository housing my midterm assessment for the TAMUSA Machine Learning class.
* This is model created by me, Noah Sizemore.
* The following information is the functioning portion. The non-functional, or prior, version used for the testing is included in the "legacy" folder.

## Overview

Assume you are working as a data scientist for the AlamoGreat Independent School District (AISD). AISD is adopting a data-driven approach to improving student performance and has two goals. First, it wants a model that predicts each student's performance, so that staff can reach out early to students who may need help. Second, it wants to know which factors are most closely associated with a student's overall performance.

In this project, you will predict each student's Performance Index from study habits and prior scores, then identify the three most important features. The project has three parts, covering data preparation, a from-scratch implementation of linear regression with gradient descent, and a computational analysis of feature importance.

By the end of this project, you will be able to do the following.

* Derive and implement gradient descent for a multi-feature linear model.
* Clean messy data and build a reproducible, leak-free pipeline.
* Predict students' Performance Index on a held-out test set.
* Measure feature importance computationally and support a claim with more than one kind of evidence.

## Part 1: The Data

You are given two files. `train.csv` contains the features and the target. `test_features.csv` contains the features and an `ID` column, but no target. Develop and select your model using `train.csv` only. Then apply your final model to `test_features.csv` and submit the predicted Performance Index for every student (see Part 2 for the file format).

| Column | Description |
| :--- | :--- |
| Hours Studied | Hours spent studying |
| Weekly Study Hours | Self-reported total study hours per week |
| Previous Scores | Score on previous exams (0 to 100) |
| Extracurricular Activities | 1 = participates, 0 = does not |
| Sleep Hours | Average hours of sleep per night |
| Sample Question Papers Practiced | Number of practice papers completed |
| Commute Minutes | Typical one-way commute to school |
| Performance Index | **Target.** Overall performance score (0 to 100). *Not included in the test file.* |

## Part 3: The Three Most Important Features

Identify the three most important features for predicting Performance Index, in ranked order. Your ranking must come from computed results, not from intuition or assumptions about what should matter. You may use scikit-learn in this part (for example, for tree-based importance); Part 2 remains NumPy only.

The table below lists methods you can use. This is not an exhaustive list. Feel free to use methods that are not mentioned in the list.

| Method | How it works | What it tells you |
| :--- | :--- | :--- |
| Permutation importance | Shuffle one feature's values in the validation set and measure how much validation RMSE increases. Repeat for each feature. | How much the trained model relies on that feature |
| Drop-one-feature retraining | Retrain the model without one feature and compare validation RMSE with the full model. Repeat for each feature. | How much predictive information is lost without that feature |
| Single-feature models | Train a model on one feature at a time and compare validation RMSE. | How much each feature can predict on its own |
| Standardized coefficients | Fit a linear model on standardized features and compare the absolute coefficient sizes. | Each feature's linear effect per standard deviation |
| Tree-based importance | Read `feature_importances_` from a random forest or gradient-boosted model. | How much each feature reduces error in the trees' splits |

### Requirements

1. Use at least two different methods from the table, or others you can justify, such as SHAP values.
2. Report the numbers. Include a table with each feature's score under each method, and base your ranking on those scores.
3. Compare the methods. State whether they produce the same ranking. Where they disagree, explain why.
4. Check stability. Repeat the analysis with at least three different random train/validation splits and report whether the ranking holds.
5. Examine the relationships. Plot each top feature against the target. Check whether any features are strongly correlated with each other, and explain how that correlation affects each method.
6. Distinguish prediction from causation. In one paragraph, state whether your results show that changing a feature would change a student's score, and what this data can and cannot support.

### Report Requirements

```python
submission = pd.DataFrame({"ID": test["ID"], "Performance Index": predictions})
submission.to_csv("predictions.csv", index=False)
