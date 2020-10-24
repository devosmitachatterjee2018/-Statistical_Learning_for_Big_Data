# Clear console CTRL+L
import os
clear = lambda: os.system('cls')  # On Windows System
clear()
 
#%%
# Importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Loading data
data = np.load('classification.npz', allow_pickle=True)

#%%
########## Exploratory data analysis ##########
# List
list = data.files
print(list)

# Data array
X_train_array =  data['X_train']
X_valid_array =  data['X_valid']
y_train_array =  data['y_train']
y_valid_array =  data['y_valid']

#%%
# Replacing arrays by dataframe
X_train_dataframe = pd.DataFrame(X_train_array)
X_valid_dataframe = pd.DataFrame(X_valid_array)
y_train_dataframe = pd.DataFrame(y_train_array)
y_valid_dataframe = pd.DataFrame(y_valid_array)

#%%
# Finding total number of rows and columns
print(X_train_dataframe.shape)
print(X_valid_dataframe.shape)
print(y_train_dataframe.shape)
print(y_valid_dataframe.shape)

#%%
print(X_train_dataframe.head(5))
print(X_train_dataframe.tail(5))

#%%
# Finding datatype
print(X_train_dataframe.dtypes)
print(X_valid_dataframe.dtypes)
print(y_train_dataframe.dtypes)
print(y_valid_dataframe.dtypes)

#%%
# Finding missing values
print(X_train_dataframe.isnull().sum())
print(y_train_dataframe.isnull().sum())
print(X_valid_dataframe.isnull().sum())
print(y_valid_dataframe.isnull().sum())

#%%
# Finding duplicate rows
duplicate_rows_X_train_dataframe = X_train_dataframe[X_train_dataframe.duplicated()]
print(duplicate_rows_X_train_dataframe.shape)
duplicate_rows_X_valid_dataframe = X_valid_dataframe[X_valid_dataframe.duplicated()]
print(duplicate_rows_X_valid_dataframe.shape)

#%%
# Finding duplicate columns
print(X_train_dataframe.columns.unique())
print(X_valid_dataframe.columns.unique())

#%%
# Plotting histogram for different categories of response variable
plt.figure(figsize=(10,6))
sns.countplot(x = 'y_train',data = data,palette = 'hls')
plt.xlabel('class labels', fontsize=12)
plt.title('Distribution of class labels Of y_train', fontsize=18)
plt.savefig('plot_count.png')

#%%
count_no_sub = len(y_train_array[y_train_array==0])
count_sub = len(y_train_array[y_train_array==1])
per = count_no_sub/count_sub
print("ratio of 0s to 1s is", per)