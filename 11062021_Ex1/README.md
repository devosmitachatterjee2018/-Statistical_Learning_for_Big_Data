## Context
The assignment is a part of the course 'Statistical_Learning_for_Big_Data', course code MVE441 at Chalmers.

## Project
A high dimensional clustering dataset (Q1_X.csv) is being provided consisting of 560 samples/observations and 974 columns/features.

The project is to determine the number of clusters in the dataset and find a method to visualise the best clustering.

## Responsibilities
- Perform an exploratory data analysis with the data in order to understand the dataset by summarizing their main characteristics, either statistically or visually.
- Implement dimensionality reduction on the data since the number of features is large relative to the number of observations.
  * Principal Component Analysis (PCA)
    - Data Standardization
    - Performing PCA on the standardized data
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
    
## Environment
Windows, Python.
