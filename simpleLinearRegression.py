#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[3]:


data=pd.read_csv("Salary_Data.csv")


# In[4]:


data.describe()


# In[5]:


X=data.iloc[:,0].values
y=data.iloc[:,1].values


# In[6]:


y=DataFrame(data,columns=['Salary'])


# In[7]:


y.shape


# In[8]:


plt.scatter(X,y)


# In[9]:


model=LinearRegression()
model.fit(X.reshape(30,1),y)


# In[10]:


model.coef_


# In[11]:


plt.scatter(X,y)
plt.plot(X.reshape(30,1),model.predict(X.reshape(30,1)))


# In[12]:


mean_absolute_error(y,model.predict(X.reshape(30,1)))


# In[13]:


model.intercept_


# In[14]:


model.score(X.reshape(30,1),y)


# In[15]:


type(data)


# In[16]:


X
  


# In[17]:


type(data[['Salary']])


# In[18]:


type(X)


# In[19]:


data['new']=10


# In[ ]:




