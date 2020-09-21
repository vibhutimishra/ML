# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#data import
dataset= pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#for missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#for encoding the names
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labely=LabelEncoder()
y=labely.fit_transform(y)
labelx=LabelEncoder()
x[:,0]=labelx.fit_transform(x[:,0])

#for dividing data to training set and test set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x, y)

#for feature scaling
from sklearn.preprocessing import StandardScaler
scalex=StandardScaler()
xtrain=scalex.fit_transform(xtrain)
xtest=scalex.transform(xtest)
