#!/usr/bin/env python
# coding: utf-8

# # Feature selection

# ## Table Of Contents
# * [Question 2: Feature selection](#chapter_1)
#     * [Section 1.1: Importing the required libraries](#section_1_1)
#     * [Section 1.2: Loading the input data csv file and importing it into dataframe](#section_1_2)
#     * [Section 1.3: Exploratory Data Analysis (EDA)](#section_1_3)
#         * [Section 1.3.1: Displaying the top 5 rows](#section_1_3_1)
#         * [Section 1.3.2: Concatenating two dataframes along column](#section_1_3_2)
#         * [Section 1.3.3: Returning True if all columns are numeric, False otherwise](#section_1_3_3)
#         * [Section 1.3.4: Displaying the summary of data](#section_1_3_4)
#         * [Section 1.3.5: Finding the total number of rows and columns](#section_1_3_5)
#         * [Section 1.3.6: Finding the duplicate rows](#section_1_3_6)
#         * [Section 1.3.7: Finding the duplicate columns](#section_1_3_7)
#         * [Section 1.3.8: Finding the missing values](#section_1_3_8)
#         * [Section 1.3.9: Finding the constant columns](#section_1_3_9)
#         * [Section 1.3.10: Dropping the constant columns](#section_1_3_10)
#     * [Section 1.4: Feature selection by Lasso regression using (A) only the features in $𝐗_{1}$](#section_1_4)
#         * [Section 1.4.1: Using only the features in $𝐗_{1}$](#section_1_4_1)
#         * [Section 1.4.2: Bootstrapping](#section_1_4_2) 
#         * [Section 1.4.3: Splitting the dataset into training and test sets](#section_1_4_3)
#         * [Section 1.4.4: Building a pipeline by a StandardScaler and the Lasso model](#section_1_4_4)
#         * [Section 1.4.5: Optimizing the hyperparameter $\alpha$ of Lasso regression](#section_1_4_5)
#         * [Section 1.4.6: Fitting the grid search to the training set ](#section_1_4_6)
#         * [Section 1.4.7: Finding the best value of the hyperparameter $\alpha$](#section_1_4_7)
#         * [Section 1.4.8: Finding the coefficients of Lasso regression](#section_1_4_8)
#         * [Section 1.4.9: Selecting the important features by Lasso regression](#section_1_4_9)
#     * [Section 1.5: Feature selection by Lasso regression using (B) both sets of features in $𝐗_{1}$ and $𝐗_{2}$ together](#section_1_5)
#         * [Section 1.5.1: Using only the features in $𝐗_{1}$](#section_1_5_1)
#         * [Section 1.5.2: Bootstrapping](#section_1_5_2) 
#         * [Section 1.5.3: Splitting the dataset into training and test sets](#section_1_5_3)
#         * [Section 1.5.4: Building a pipeline by a StandardScaler and the Lasso model](#section_1_5_4)
#         * [Section 1.5.5: Optimizing the hyperparameter $\alpha$ of Lasso regression](#section_1_5_5)
#         * [Section 1.5.6: Fitting the grid search to the training set ](#section_1_5_6)
#         * [Section 1.5.7: Finding the best value of the hyperparameter $\alpha$](#section_1_5_7)
#         * [Section 1.5.8: Finding the coefficients of Lasso regression](#section_1_5_8)
#         * [Section 1.5.9: Selecting the important features by Lasso regression](#section_1_5_9)
#     * [Section 1.6: Discussion](#section_1_6)

# ### Question 2: Feature selection <a class="anchor" id="chapter_1"></a>

# ##### Section 1.1: Importing the required libraries <a class="anchor" id="section_1_1"></a>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso


# ##### Section 1.2: Loading the input data csv file and importing it into dataframe <a class="anchor" id="section_1_2"></a>

# In[2]:


y = pd.read_csv("C:\\Users\\Acer\\Desktop\\Big Data 2021\\Q2_y.csv")
X1 = pd.read_csv("C:\\Users\\Acer\\Desktop\\Big Data 2021\\Q2_X1.csv")
X2 = pd.read_csv("C:\\Users\\Acer\\Desktop\\Big Data 2021\\Q2_X2.csv")


# An exploratory data analysis is performed with data in order to understand the dataset by summarizing their main characteristics, either statistically or visually.

# ##### Section 1.3: Exploratory Data Analysis (EDA) <a class="anchor" id="section_1_3"></a>

# ##### Section 1.3.1: Displaying the top 5 rows <a class="anchor" id="section_1_3_1"></a>

# In[36]:


print(y.head())
print(X1.head())
print(X2.head())


# ##### Section 1.3.2: Concatenating two dataframes along column <a class="anchor" id="section_1_3_2"></a>

# In[4]:


X1_X2 = pd.concat([X1, X2], axis=1)
print(X1_X2)


# ##### Section 1.3.3: Returning True if all columns are numeric, False otherwise <a class="anchor" id="section_1_3_3"></a>

# In[5]:


print(y.shape[1] == y.select_dtypes(include=np.number).shape[1])
print(X1.shape[1] == X1.select_dtypes(include=np.number).shape[1])
print(X1_X2.shape[1] == X1_X2.select_dtypes(include=np.number).shape[1])


# The above result shows that all columns are numeric.

# ##### Section 1.3.4: Displaying the summary of data <a class="anchor" id="section_1_3_4"></a>

# Checking the properties of the numeric features.

# In[6]:


print(X1.describe())
print(X1_X2.describe())


# ##### Section 1.3.5: Finding the total number of rows and columns <a class="anchor" id="section_1_3_5"></a>

# In[7]:


print(y.shape)
print(X1.shape)
print(X1_X2.shape)


# The above result shows that the dataset X = (X1,X2) is high dimensional that is, number of columns/features greater than number of rows/observations.

# ##### Section 1.3.6: Finding the duplicate rows <a class="anchor" id="section_1_3_6"></a>

# In[8]:


print(X1.duplicated().sum())
print(X1_X2.duplicated().sum())


# The above result shows that there is no duplicate row in X1 and X = (X1,X2).

# ##### Section 1.3.7: Finding the duplicate columns <a class="anchor" id="section_1_3_7"></a>

# In[9]:


print(X1.columns.duplicated().sum())
print(X1_X2.columns.duplicated().sum())


# The above result shows that there is no duplicate column in X1 and X = (X1,X2).

# ##### Section 1.3.8: Finding the missing values <a class="anchor" id="section_1_3_8"></a>

# In[10]:


print(y.isnull().sum().sum())
print(X1.isnull().sum().sum())
print(X1_X2.isnull().sum().sum())


# The above result shows that there is no missing value in y, X1 and X = (X1,X2).

# ##### Section 1.3.9: Finding the constant columns <a class="anchor" id="section_1_3_9"></a>

# In[11]:


print(X1.columns[X1.nunique() <= 1])
print(X1_X2.columns[X1_X2.nunique() <= 1])


# The above result shows that there is no constant column in X1 and X = (X1,X2).

# ##### Section 1.3.10: Finding the feature names <a class="anchor" id="section_1_3_10"></a>

# In[12]:


features = list(X1.columns)
print(features)
features2 = list(X1_X2.columns)
print(features2)


# ##### Section 1.4: Feature selection by Lasso regression  using (A) only the features in $𝐗_{1}$ <a class="anchor" id="section_1_4"></a>

# ##### Section 1.4.1: Using only the features in $𝐗_{1}$ <a class="anchor" id="section_1_4_1"></a>

# In[13]:


A = X1
b = y


# In[14]:


X_train_prev, X_test_prev, y_train_prev, y_test_prev = train_test_split(X1,y,test_size=0.3, random_state=0)
model_prev = LinearRegression()
model_prev.fit(X_train_prev, y_train_prev)
y_predicted_prev = model_prev.predict(X_test_prev)

fig, ax = plt.subplots()
ax.scatter(y_predicted_prev, y_test_prev, edgecolors=(0, 0, 1))
ax.plot([y_test_prev.min(), y_test_prev.max()], [y_test_prev.min(), y_test_prev.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.savefig('C:\\Users\\Acer\\Desktop\\Big Data 2021\\ActualvsPredicted_A_prev.png')
plt.show()

mae_prev = metrics.mean_absolute_error(y_test_prev, y_predicted_prev)
mse_prev = metrics.mean_squared_error(y_test_prev, y_predicted_prev)
r2_prev = metrics.r2_score(y_test_prev, y_predicted_prev)
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae_prev))
print('MSE is {}'.format(mse_prev))
print('R2 score is {}'.format(r2_prev))


# ##### Section 1.4.2: Bootstrapping <a class="anchor" id="section_1_4_2"></a>

# In[15]:


boot_index_replications = np.array([np.random.choice(len(A),int(np.ceil(0.8*len(A))),replace = True)for _ in range(1000)])
boot_index = np.mean(boot_index_replications, axis=1)


# ##### Section 1.4.3: Splitting the dataset into training and test sets <a class="anchor" id="section_1_4_3"></a>

# The dataset is split into training and test sets. But all the calculations are performed on the training set only. 

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(A.iloc[boot_index], b.iloc[boot_index], test_size=0.30)


# ##### Section 1.4.4: Building a pipeline by data standardization and Lasso model <a class="anchor" id="section_1_4_4"></a>

# The pipeline is made by the StandardScaler and the Lasso object.

# Data standardization of a feature means to scale the observations of the feature with mean 0 and standard deviation 1 given by the formula $$Z = \frac{X - mean(X)}{sd(X)}$$.

# In[17]:


pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
                    ])


# ##### Section 1.4.5: Optimizing the hyperparameter  $\alpha$ of Lasso regression <a class="anchor" id="section_1_4_5"></a>

# The hyperparameter $\alpha$ of Lasso regression is optimized using the GridSearchCV object in the following way. 
# 
# 1. Several $\alpha$ values are tested from 0.1 to 10 with step 0.1. 
# 
# 2. For each $\alpha$ value, the average value of the mean squared error is calculated in a 5-folds cross-validation.
# 
# 3. Select the value of $\alpha$ that minimizes such performance metrics. 

# In[18]:


search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )


# ##### Section 1.4.6: Fitting the grid search to the training set <a class="anchor" id="section_1_4_6"></a>

# In[19]:


search.fit(X_train,y_train)


# ##### Section 1.4.7: Finding the best value of the hyperparameter $\alpha$ <a class="anchor" id="section_1_4_7"></a>

# In[20]:


search.best_params_


# ##### Section 1.4.8: Finding the coefficients of Lasso regression <a class="anchor" id="section_1_4_8"></a>

# In[21]:


coefficients = search.best_estimator_.named_steps['model'].coef_
print(coefficients)


# ##### Section 1.4.9: Selecting the important features by Lasso regression <a class="anchor" id="section_1_4_9"></a>

# The importance of a feature is the absolute value of its coefficient.

# In[22]:


importance = np.abs(coefficients)
print(len(np.array(features)[importance > 0]))
print(np.array(features)[importance > 0])
print(len(np.array(features)[importance == 0]))
print(np.array(features)[importance == 0])


# In[23]:


X1_new = X1[list(np.array(features)[importance > 0])]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X1_new,y,test_size=0.3, random_state=0)
model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)
y_predicted_new = model_new.predict(X_test_new)

fig, ax = plt.subplots()
ax.scatter(y_predicted_new, y_test_new, edgecolors=(0, 0, 1))
ax.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.savefig('C:\\Users\\Acer\\Desktop\\Big Data 2021\\ActualvsPredicted_A_new.png')

plt.show()

mae_new = metrics.mean_absolute_error(y_test_new, y_predicted_new)
mse_new = metrics.mean_squared_error(y_test_new, y_predicted_new)
r2_new = metrics.r2_score(y_test_new, y_predicted_new)
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae_new))
print('MSE is {}'.format(mse_new))
print('R2 score is {}'.format(r2_new))


# ##### Section 1.5: Feature selection by Lasso regression using (B) both sets of features in $𝐗_{1}$ and $𝐗_{2}$ together <a class="anchor" id="section_1_5"></a>

# ##### Section 1.5.1: Using both sets of features in $𝐗_{1}$ and $𝐗_{2}$ together <a class="anchor" id="section_1_5_1"></a>

# In[24]:


A2 = X1_X2
b = y


# In[25]:


X_train2_prev, X_test2_prev, y_train2_prev, y_test2_prev = train_test_split(X1_X2,y,test_size=0.3, random_state=0)
model2_prev = LinearRegression()
model2_prev.fit(X_train2_prev, y_train2_prev)
y_predicted2_prev = model2_prev.predict(X_test2_prev)

fig, ax = plt.subplots()
ax.scatter(y_predicted2_prev, y_test2_prev, edgecolors=(0, 0, 1))
ax.plot([y_test2_prev.min(), y_test2_prev.max()], [y_test2_prev.min(), y_test2_prev.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.savefig('C:\\Users\\Acer\\Desktop\\Big Data 2021\\ActualvsPredicted_B_prev.png')
plt.show()

mae2_prev = metrics.mean_absolute_error(y_test2_prev, y_predicted2_prev)
mse2_prev = metrics.mean_squared_error(y_test2_prev, y_predicted2_prev)
r22_prev = metrics.r2_score(y_test2_prev, y_predicted2_prev)
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae2_prev))
print('MSE is {}'.format(mse2_prev))
print('R2 score is {}'.format(r22_prev))


# ##### Section 1.5.2: Bootstrapping <a class="anchor" id="section_1_5_2"></a>

# In[26]:


boot_index_replications2 = np.array([np.random.choice(len(A2),int(np.ceil(0.8*len(A2))),replace = True)for _ in range(1000)])
boot_index2 = np.mean(boot_index_replications2, axis=1)


# ##### Section 1.5.3: Splitting the dataset into training and test sets <a class="anchor" id="section_1_5_3"></a>

# In[27]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(A2.iloc[boot_index2], b.iloc[boot_index2], test_size=0.30)


# ##### Section 1.5.4: Building a pipeline by data standardization and Lasso model <a class="anchor" id="section_1_5_4"></a>

# In[28]:


pipeline2 = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])


# ##### Section 1.5.5: Optimizing the hyperparameter  $\alpha$ of Lasso regression <a class="anchor" id="section_1_5_5"></a>

# In[29]:


search2 = GridSearchCV(pipeline2,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )


# ##### Section 1.5.6: Fitting the grid search to the training set <a class="anchor" id="section_1_5_6"></a>

# In[30]:


search2.fit(X_train2,y_train2)


# ##### Section 1.5.7: Finding the best value of the hyperparameter $\alpha$ <a class="anchor" id="section_1_5_7"></a>

# In[31]:


search2.best_params_


# ##### Section 1.5.8: Finding the coefficients of Lasso regression <a class="anchor" id="section_1_5_8"></a>

# In[32]:


coefficients2 = search2.best_estimator_.named_steps['model'].coef_


# ##### Section 1.5.9: Selecting the important features by Lasso regression <a class="anchor" id="section_1_5_9"></a>

# In[33]:


importance2 = np.abs(coefficients2)
print(len(np.array(features2)[importance2 > 0]))
print(np.array(features2)[importance2 > 0])
print(len(np.array(features2)[importance2 == 0]))
print(np.array(features2)[importance2 == 0])


# In[34]:


X1_X2_new = X1_X2[list(np.array(features2)[importance2 > 0])]

X_train2_new, X_test2_new, y_train2_new, y_test2_new = train_test_split(X1_X2_new,y,test_size=0.3, random_state=0)
model2_new = LinearRegression()
model2_new.fit(X_train2_new, y_train2_new)
y_predicted2_new = model2_new.predict(X_test2_new)

fig, ax = plt.subplots()
ax.scatter(y_predicted2_new, y_test2_new, edgecolors=(0, 0, 1))
ax.plot([y_test2_new.min(), y_test2_new.max()], [y_test2_new.min(), y_test2_new.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Actual vs. Predicted')
plt.savefig('C:\\Users\\Acer\\Desktop\\Big Data 2021\\ActualvsPredicted_B_new.png')
plt.show()

mae2_new = metrics.mean_absolute_error(y_test2_new, y_predicted2_new)
mse2_new = metrics.mean_squared_error(y_test2_new, y_predicted2_new)
r22_new = metrics.r2_score(y_test2_new, y_predicted2_new)
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae2_new))
print('MSE is {}'.format(mse2_new))
print('R2 score is {}'.format(r22_new))


# ##### Section 1.6: Discussion <a class="anchor" id="section_1_6"></a>

# 1. From exploratory data analysis of the given datasets, t is found that $X$=($X_{1}$,$X_{2}$) is high dimensional dataset that is, number of features greater than number of observations.
# 
# 2. Lasso regression is a popular feature selection technique for high dimensional dataset. Lasso uses a penalization method which reduces the chance of overestimating a regression coefficient.
# 
# 3. Considering the computed regression metrics like the mean absolute error, the mean squared error, and the R-squared, it is concluded that the linear regression model is improved after feature selection in both the cases.
# 
