## Context
The assignment is a part of the course 'Statistical_Learning_for_Big_Data', course code MVE441 at Chalmers.

## Project
A regression dataset is being provided containing a response vector y of order n * 1 (Q2_y.csv) and two sets of features X1 of order n * p1  (Q2_X1.csv) and X2 of order n * p2 (Q2_X2.csv).

The project is to determine the most important features for the response y using (A) only the features in X1 and then (B) both sets of features in X1 and X2 together X = (X1,X2) of order n * (p1+p2), justify the feature selection in each case, and explain the difference between the results obtained from (A) and (B).

## Responsibilities
- Perform an exploratory data analysis in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
  *  Data size
  *  Data type
  *  Missing data
  *  Duplicate data
  *  Constant columns
  
- Apply feature selection technique using (A) only the features in X1 and (B) both sets of features in X1 and X2 together X = (X1,X2).
  * Lasso Regression
    - Do bootstrapping.
    - Build a pipeline by data standardization and Lasso model.
    - Optimize the hyperparameter alpha of Lasso regression.
    - Select the important features by Lasso regression.
    - Evaluate the model using regression metrics such as the mean absolute error, the mean squared error, and the R-squared before and after feature selection.

## Environment
Windows, Python.
