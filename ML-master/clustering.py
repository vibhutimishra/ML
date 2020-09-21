# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:35:19 2020

@author: Vibuthi mishra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using elbow method to find number of clusteres
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init = "k-means++",max_iter=300,n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("number")
plt.ylabel("wcss")
plt.show()

#applying k means cluster
kmeans=KMeans(n_clusters=5,init="k-means++")
ykmeans=kmeans.fit_predict(X)

#visualize the cluster
plt.scatter(X[ykmeans==0,0],X[ykmeans==0,1],s=50,c="red")
plt.scatter(X[ykmeans==1,0],X[ykmeans==1,1],s=50,c="blue")
plt.scatter(X[ykmeans==2,0],X[ykmeans==2,1],s=50,c="black")
plt.scatter(X[ykmeans==3,0],X[ykmeans==3,1],s=50,c="green")
plt.scatter(X[ykmeans==4,0],X[ykmeans==4,1],s=50,c="orange")
plt.xlabel("anual income")
plt.ylabel("scores")
plt.show()