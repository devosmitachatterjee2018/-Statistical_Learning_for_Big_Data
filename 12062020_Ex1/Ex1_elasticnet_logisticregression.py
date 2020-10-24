# Clear console CTRL+L
import os
clear = lambda: os.system('cls')  # On Windows System
clear()
 
#%%
# Importing libraries
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

#%%
# Load data
data = np.load('classification.npz', allow_pickle=True)

list = data.files
print(list)

# Data array
X_train_array =  data['X_train']
X_valid_array =  data['X_valid']
y_train_array =  data['y_train']
y_valid_array =  data['y_valid']

# Replacing arrays by dataframe
X_train_dataframe = pd.DataFrame(X_train_array)
X_valid_dataframe = pd.DataFrame(X_valid_array)
y_train_dataframe = pd.DataFrame(y_train_array)
y_valid_dataframe = pd.DataFrame(y_valid_array)

#%%
# Data standardization
X_train_array = StandardScaler().fit_transform(X_train_array)
X_valid_array = StandardScaler().fit_transform(X_valid_array)

#%%
# define models and parameters
solver = ['saga', 'liblinear']
penalty = ['elasticnet']
C = [ 0.001, 0.01, 0.1, 1]
# define grid search
grid = dict(solver=solver,penalty=penalty,C=C)
bestparameterslogisticregressionmodel = GridSearchCV(estimator=LogisticRegression(random_state=0,l1_ratio=0.8), param_grid=grid,
                    verbose=1,
                    n_jobs=-1)
bestparameterslogisticregressionmodel.fit(X_train_array, y_train_array)
# summarize results
print("Best parameters are %s" % (bestparameterslogisticregressionmodel.best_params_))

#%%
C = [ 0.001, 0.01, 0.1, 1]

for c in C:
    clf = LogisticRegression(random_state=0,l1_ratio=0.8, penalty='elasticnet', C=c, solver='saga')
    clf.fit(X_train_array, y_train_array)
    print('C:', c)
    print('Training accuracy:', clf.score(X_train_array, y_train_array))
    print('Test accuracy:', clf.score(X_valid_array, y_valid_array))
    print('')
    
#%%
########## Before feature selection ##########
# Define model
logisticregressionmodel = LogisticRegression(random_state=0,l1_ratio=0.8, penalty='elasticnet', C=0.1, solver='saga')
# Fit the model
logisticregressionmodel.fit(X_train_array, y_train_array)
# Confusion matrix
y_pred_array = logisticregressionmodel.predict(X_valid_array)
accuracyscore = metrics.accuracy_score(y_valid_array, y_pred_array)
print(accuracyscore)
# Confusion matrix is used to describe the performance of a classification model
confusionmatrix = metrics.confusion_matrix(y_valid_array, y_pred_array)
print(confusionmatrix)
print(metrics.classification_report(y_valid_array, y_pred_array))
plt.figure(figsize=(10,6))
sns.heatmap(confusionmatrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
all_sample_title = 'Confusion matrix for elastic net-logistic regression\n Accuracy Score: {0}'.format(accuracyscore)
plt.title(all_sample_title, fontsize = 18)
plt.savefig('plot_beforeselection_elasticnet_confusionmatrix.png')

#%%
# Receiver operating characteristic (ROC) 
logit_roc_auc = roc_auc_score(y_valid_array, y_pred_array)
print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_valid_array, logisticregressionmodel.predict_proba(X_valid_array)[:,1])
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label='Elastic net-logistic regression (AUC = %f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate', fontsize = 12)
plt.ylabel('True positive rate', fontsize = 12)
all_sample_title = 'ROC curve for elastic net-logistic regression\n ROC AUC Score: {0}'.format(logit_roc_auc)
plt.title(all_sample_title, fontsize = 18)
plt.legend(loc="lower right")
plt.savefig('plot_beforeselection_elasticnet_ROCcurve.png')
plt.show()

#%%
########## Feature importance ##########
# Get importance
plt.figure(figsize=(10,6))
importance = logisticregressionmodel.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance scores', fontsize=12)
plt.title('Bar chart of elastic net-logistic regression coefficients as feature importance scores', fontsize=18)
plt.savefig('plot_elasticnet_featureimportancescores.png')
plt.show()

#%%
########## Feature selection ##########
#Selecting features using Lasso regularisation using SelectFromModel
sel_ = SelectFromModel(logisticregressionmodel)
sel_.fit(X_train_array, y_train_array)

#%%
#Make a list of with the selected features
selected_feat = X_train_dataframe.columns[(sel_.get_support())]
print('total features: {}'.format((X_train_dataframe.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))

#%%
#Removing the features from training an test set
X_train_selected = sel_.transform(X_train_array)
X_valid_selected = sel_.transform(X_valid_array)
print(X_train_selected.shape)
print(X_valid_selected.shape)

#%%
########## After feature selection ##########
# Evaluating the model
# Measuring methods' performance
logisticregressionmodel.fit(X_train_selected, y_train_array)
# Confusion matrix
y_pred_array = logisticregressionmodel.predict(X_valid_selected)
accuracyscore = metrics.accuracy_score(y_valid_array, y_pred_array)
print(accuracyscore)
# Confusion matrix is used to describe the performance of a classification model
confusionmatrix = metrics.confusion_matrix(y_valid_array, y_pred_array)
print(confusionmatrix)
print(metrics.classification_report(y_valid_array, y_pred_array))
plt.figure(figsize=(10,6))
sns.heatmap(confusionmatrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
all_sample_title = 'Confusion matrix for elastic net-logistic regression\n Accuracy Score: {0}'.format(accuracyscore)
plt.title(all_sample_title, fontsize = 18)
plt.savefig('plot_afterselection_elasticnet_confusionmatrix.png')

#%%
# Receiver operating characteristic (ROC) 
logit_roc_auc = roc_auc_score(y_valid_array, y_pred_array)
print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_valid_array, logisticregressionmodel.predict_proba(X_valid_selected)[:,1])
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label='Elastic net-logistic regression (AUC = %f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate', fontsize = 12)
plt.ylabel('True positive rate', fontsize = 12)
all_sample_title = 'ROC curve for elastic net-logistic regression\n ROC AUC Score: {0}'.format(logit_roc_auc)
plt.title(all_sample_title, fontsize = 18)
plt.legend(loc="lower right")
plt.savefig('plot_afterselection_elasticnet_ROCcurve.png')
plt.show()

