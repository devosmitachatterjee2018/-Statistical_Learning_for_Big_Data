## Context
The assignment is a part of the course 'Statistical_Learning_for_Big_Data', course code MVE441 at Chalmers.

## Project 1: Clustering
We are provided with a clustering dataset (Q1_X.csv) containing 560 samples/observations and 974 columns/features.

The project is to determine the number of clusters in the dataset and find a way to visualise the best clustering.

## Responsibilities for project 1
- Perform an exploratory data analysis with the data in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
- Implement dimensionality reduction on the data since the number of features is large relative to the number of observations.
  * Principal Component Analysis
- Use different clustering algorithms on the obtained principal components to obtain high quality clusters in the data.
  * K-means
  * Gaussian Mixture Models (GMM)
  * Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

## Project 2: Feature selection
We are provided with a regression dataset containing a response vector 𝐲 of order n * 1 (Q2_y.csv) and two sets of features 𝐗1 of order n * p1  (Q2_X1.csv) and 𝐗2 of order n * p2 (Q2_X2.csv).

The project is to determine the most important features for the response 𝐲 using (A) only the features in 𝐗1 and then (B) both sets of features in 𝐗1 and 𝐗2 together (𝐗 = (𝐗1,𝐗2) of order n * (𝑝1+𝑝2), justify the feature selection in each case, and explain the difference between the results obtained from (A) and (B).

## Responsibilities for project 2
- Perform an exploratory data analysis.
- Apply feature selection technique.
  * Lasso Regression
- Evaluate the model using regression metrics such as the mean absolute error, the mean squared error, and the R-squared before and after feature selection.

## Environment
Windows, Python.
