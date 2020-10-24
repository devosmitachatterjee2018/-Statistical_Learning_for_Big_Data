# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 00:07:28 2020

@author: a347001
"""


# Clear console CTRL+L
import os
clear = lambda: os.system('cls')  # On Windows System
clear()
 
#%%
# Importing libraries
import numpy as np 
import pandas as pd
#%%
# Load data
data = np.load('august2020-exercise1.npz', allow_pickle=True)

list = data.files

# Data array
X_lr_small_train_array =  data['X_lr_small_train']
y_lr_small_train_array =  data['y_lr_small_train']
X_lr_big_train_array =  data['X_lr_big_train']
y_lr_big_train_array =  data['y_lr_big_train']

print(data['X_hr_small_train'])
print(data['y_hr_small_train'])
print(data['X_hr_big_train']) 
print(data['y_hr_big_train'])
print(data['y_lr_test'])
print(data['X_lr_test'])
print(data['X_hr_test'])
print(data['y_hr_test'])

