#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn.datasets import load_boston
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

#loading boston data
dataset=load_boston()
type(dataset)
dir(dataset)

# ## data points and features
dataset.data.shape
# ### data exploration with pandas dataframe
data=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)
data['target']=dataset.target

data.head()
data.count()


# ## cleaning data -missing values
pd.isnull(data).any()
# ## visualization
plt.figure(figsize=[10,6])
plt.hist(data['target'],bins=50,ec='black')
plt.xlabel("prices")
plt.ylabel("No. of houses")

plt.figure(figsize=[10,6])
sns.distplot(data['target'])
plt.hist(data['RAD'])


freq=data['RAD'].value_counts()
freq
freq2=data['CHAS'].value_counts()
freq2

# ## descriptive analysis
data['target'].min()
data.min()
data.mean()

#  #### correlation 
data['target'].corr(data['RM'])
data.corr()

# ### multiple linear regression model
prices=data['target']
features=data.drop('target',axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(features,prices,random_state=42,test_size=0.2)
regr = LinearRegression()
regr.fit(xtrain,xtest)

regr.intercept_

pd.DataFrame(data=regr.coef_,index=xtrain.columns)