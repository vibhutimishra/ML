#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data=pd.read_csv("Salary_Data.csv")
data.describe()

X=data.iloc[:,0].values
y=data.iloc[:,1].values

y=DataFrame(data,columns=['Salary'])
y.shape

# building model
plt.scatter(X,y)
model=LinearRegression()
model.fit(X.reshape(30,1),y)
model.coef_

#plot of X and y
plt.scatter(X,y)
plt.plot(X.reshape(30,1),model.predict(X.reshape(30,1)))

mean_absolute_error(y,model.predict(X.reshape(30,1)))
model.intercept_
model.score(X.reshape(30,1),y)
type(data)
X
type(data[['Salary']])
type(X)
data['new']=10
