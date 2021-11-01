## Context
The assignment is a part of the Course 'Statistical Learning for Big Data', Course Code MVE441 at Chalmers.

## Project 1
A high dimensional clustering dataset (Q1_X.csv) is being provided consisting of 560 samples/observations and 974 columns/features.

The project is to determine the number of clusters in the dataset and find a method to visualise the best clustering.

## Responsibilities for project 1
- Perform an exploratory data analysis with the data in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
  *  Data size
  *  Data type
  *  Missing data
  *  Duplicate data
  *  Constant columns
  
- Implement dimensionality reduction on the data since the number of features is large relative to the number of observations.
  * Principal Component Analysis (PCA)
    - Standardize the data.
    - Performing PCA on the standardized data.
- Use different clustering algorithms to obtain high quality clusters in the data.
  * K-means
    - Find an optimal k using Elbow method.
    - Find an optimal k using Silhouette method.
    - Cluster using k-means based on principal components.
  * Gaussian Mixture Models (GMM)
    - Minimize the Bayesian Information Criterion (BIC) for optimal number of clusters k.
    - Minimize the Akaike Information Criterion (AIC) for optimal number of clusters k.
    - Cluster using GMM based on principal components.
  * Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    - Find the parameter epsilon by plotting k-distance graph.
    - Cluster using DBCAN based on principal components.
   
## Project 2
A regression dataset is being provided containing a response vector y of order n * 1 (Q2_y.csv) and two sets of features X1 of order n * p1  (Q2_X1.csv) and X2 of order n * p2 (Q2_X2.csv).

The project is to determine the most important features for the response y using (A) only the features in X1 and then (B) both sets of features in X1 and X2 together X = (X1,X2) of order n * (p1+p2), justify the feature selection in each case, and explain the difference between the results obtained from (A) and (B).

## Responsibilities for project 2
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
