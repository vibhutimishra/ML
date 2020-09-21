# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:14:53 2020

@author: Vibuthi mishra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(random_state=0,n_estimators=1000)
regressor.fit(x,y)



xgrid=np.arange(min(x),max(x),0.1)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(x, y)
plt.plot(xgrid,regressor.predict(xgrid))
plt.show()
