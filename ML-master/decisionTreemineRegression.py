# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 23:19:03 2020

@author: Vibuthi mishra
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

print(min(x))
#plot
xgrid=np.arange(min(x),max(x),0.05)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(x,y)
plt.plot(xgrid,regressor.predict(xgrid))
plt.xlabel("level")
plt.ylabel("salaries")
plt.show()


print(xgrid)