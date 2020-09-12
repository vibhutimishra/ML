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


# In[25]:


dataset=load_boston()


# In[26]:


type(dataset)


# In[27]:


dir(dataset)


# ## data points and features

# In[28]:


dataset.data.shape


# ### data exploration with pandas dataframe

# In[29]:


data=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)
data['target']=dataset.target


# In[30]:


data.head()


# In[31]:


data.count()


# ## cleaning data -missing values

# In[32]:


pd.isnull(data).any()


# ## visualization

# In[33]:


plt.figure(figsize=[10,6])
plt.hist(data['target'],bins=50,ec='black')
plt.xlabel("prices")
plt.ylabel("No. of houses")


# In[34]:


plt.figure(figsize=[10,6])
sns.distplot(data['target'])


# In[35]:


plt.hist(data['RAD'])


# In[36]:


freq=data['RAD'].value_counts()
freq


# In[37]:


freq2=data['CHAS'].value_counts()
freq2


# ## descriptive analysis
# 

# In[38]:


data['target'].min()


# In[39]:


data.min()


# In[40]:


data.mean()


#  #### correlation 

# In[41]:


data['target'].corr(data['RM'])


# In[42]:


data.corr()


# ### multiple linear regression

# In[43]:


prices=data['target']
features=data.drop('target',axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(features,prices,random_state=42,test_size=0.2)


# In[45]:


regr = LinearRegression()
regr.fit(xtrain,xtest)


# In[46]:


regr.intercept_


# In[48]:


pd.DataFrame(data=regr.coef_,index=xtrain.columns)


# In[ ]:





# In[ ]:




