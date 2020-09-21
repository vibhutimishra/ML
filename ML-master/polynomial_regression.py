# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 23:07:54 2020

@author: Vibuthi mishra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv("Position_Salaries.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#creating linear regression
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(x, y)

#creating polynomial regression
from sklearn.preprocessing import PolynomialFeatures
preg=PolynomialFeatures(degree=4)
xpoly=preg.fit_transform(x)

#creating linear model for polynomial data
lreg2=LinearRegression()
lreg2.fit(xpoly,y)

#visualising the linear model
plt.scatter(x, y)
plt.plot(x,lreg.predict(x))
plt.title("linear_prediction")
plt.xlabel("level")
plt.ylabel("salaries_linear")
plt.show()

#visualising the polynomial 
plt.scatter(x,y)
plt.plot(x,lreg2.predict(xpoly))
plt.title("polynomail_prediction")
plt.ylabel("salaries")
plt.xlabel("level")
plt.show()



