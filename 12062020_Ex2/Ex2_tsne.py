# Clear console
import os
clear = lambda: os.system('cls')
clear()

#%%
# Importing libraries
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Loading data
data = np.load('clustering.npz', allow_pickle=True)
list = data.files
print(list)

#%%
# Data array
X_train_array =  data['X']
print(X_train_array)

#%%
##### Exploratory data analysis #####
# Replacing arrays by dataframe
X_train_dataframe = pd.DataFrame(X_train_array)
print(X_train_dataframe)

#%%
# Summary
print(X_train_dataframe.describe())

#%%
# Finding datatype
print(X_train_dataframe.dtypes)

#%%
# Finding total number of rows and columns
print(X_train_dataframe.shape)

#%%
# Finding duplicate rows
duplicate_rows_dataframe = X_train_dataframe[X_train_dataframe.duplicated()]
print(duplicate_rows_dataframe.shape)

#%%
# Finding duplicate columns
print(X_train_dataframe.columns.unique())

#%%
# Finding missing values
print(X_train_dataframe.isnull().sum())

#%% 
# Data Normalization
X_train_array = Normalizer().fit_transform(X_train_array)

# TSNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(X_train_array)
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1]})

plt.figure(figsize=(10,6))
# Finding the number of clusters
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k,random_state=0)
    km = km.fit(tsne_df)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bo-',3,Sum_of_squared_distances[3]+1500, 'ro')
plt.annotate('Optimal number of clusters k=3', xy=(3,Sum_of_squared_distances[3]), xytext=(3.5, 25000),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Number of clusters (k)', fontsize=12)
plt.ylabel('Within Cluster Sum of Squares', fontsize=12)
plt.title('Elbow method for optimal k', fontsize=18)
plt.savefig('plot_ElbowMethodForOptimalk.png')
plt.show()

#%%
# KMeans
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(tsne_df)
y_kmeans = kmeans.predict(tsne_df)
C = kmeans.cluster_centers_

plt.figure(figsize=(10,6))
sns.scatterplot(
    x="X", y="Y", hue = y_kmeans,
    palette=sns.color_palette("hls", 3),
              legend='full',
              data=tsne_df);
plt.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=100)
plt.title('Kmeans clustering with TSNE dimensionality reduction', fontsize=18)
plt.xlabel('TSNE dimension 1', fontsize=12)
plt.ylabel('TSNE dimension 2', fontsize=12)
plt.savefig('plot_KmeansClusteringWithTSNEDimensionalityReduction.png')

#%%
# Find five variables that are most indicative of each found cluster.
kmeans.fit(X_train_array)
print(kmeans.labels_)
res=kmeans.__dict__
print(res)

#featureslist_cluster1 = res['cluster_centers_'][0]
#featureslist_cluster2 = res['cluster_centers_'][1]
#featureslist_cluster3 = res['cluster_centers_'][2]
#plt.figure(figsize=(10,6))
#plt.plot(np.arange(1,729, 1), featureslist_cluster1, 'ro',np.arange(1,729, 1), featureslist_cluster2, 'g^',np.arange(1,729, 1), featureslist_cluster3, 'b.')
#plt.title('Feature importance plot', fontsize=18)
#plt.xlabel('Features', fontsize=12)
#plt.ylabel('Cluster centers', fontsize=12)
#plt.savefig('plot_FeatureImportancePlot.png')
#%%
C1 = np.zeros((728))
C2 = np.zeros((728))
C3 = np.zeros((728))
C1C2 = np.zeros((728))
C3C1 = np.zeros((728))
C1C2C3C1 = np.zeros((728))
for i in range(0, 727, 1):
    C1[i] = res['cluster_centers_'][0][i]
    C2[i] = res['cluster_centers_'][1][i]
    C3[i] = res['cluster_centers_'][2][i]
    C1C2[i] = np.absolute((C1[i]-C2[i]))
    C3C1[i] = np.absolute((C3[i]-C1[i]))
    C1C2C3C1[i] = C1C2[i] + C3C1[i]
    
featureimportance_df1 = pd.DataFrame({'Features': np.arange(1,729, 1),'C1': C1,'C2': C2,'C1C2': C1C2,'C3C1': C3C1,'C1C2C3C1': C1C2C3C1})
featureimportance_df1_sorted = featureimportance_df1.sort_values(by='C1C2C3C1', ascending = False).iloc[:,0:6]
print(featureimportance_df1_sorted)
 
#%%
C1 = np.zeros((728))
C2 = np.zeros((728))
C3 = np.zeros((728))
C1C2 = np.zeros((728))
C2C3 = np.zeros((728))
C1C2C2C3 = np.zeros((728))
for i in range(0, 727, 1):
    C1[i] = res['cluster_centers_'][0][i]
    C2[i] = res['cluster_centers_'][1][i]
    C3[i] = res['cluster_centers_'][2][i]
    C1C2[i] = np.absolute((C1[i]-C2[i]))
    C2C3[i] = np.absolute((C2[i]-C3[i]))
    C1C2C2C3[i] = C1C2[i] + C2C3[i]
    
featureimportance_df2 = pd.DataFrame({'Features': np.arange(1,729, 1),'C1': C1,'C2': C2,'C1C2': C1C2,'C2C3': C3C1,'C1C2C2C3': C1C2C2C3})
featureimportance_df2_sorted = featureimportance_df2.sort_values(by='C1C2C2C3', ascending = False).iloc[:,0:6]
print(featureimportance_df2_sorted)

#%%
C1 = np.zeros((728))
C2 = np.zeros((728))
C3 = np.zeros((728))
C2C3 = np.zeros((728))
C3C1 = np.zeros((728))
C2C3C3C1 = np.zeros((728))
for i in range(0, 727, 1):
    C1[i] = res['cluster_centers_'][0][i]
    C2[i] = res['cluster_centers_'][1][i]
    C3[i] = res['cluster_centers_'][2][i]
    C2C3[i] = np.absolute((C2[i]-C3[i]))
    C3C1[i] = np.absolute((C3[i]-C1[i]))
    C2C3C3C1[i] =  C2C3[i] + C3C1[i]
    
featureimportance_df3 = pd.DataFrame({'Features': np.arange(1,729, 1),'C1': C1,'C2': C2,'C2C3': C3C1,'C3C1': C3C1, 'C2C3C3C1': C2C3C3C1})
featureimportance_df3_sorted = featureimportance_df3.sort_values(by='C2C3C3C1', ascending = False).iloc[:,0:6]
print(featureimportance_df3_sorted)