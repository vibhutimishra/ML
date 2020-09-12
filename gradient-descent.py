#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[5]:


def cubic(x):
    return x**3+x**2+x+1
def dfx(x):
    return 3*x*x+2*x+1


# In[6]:


x_=np.linspace(start=-2,stop=3,num=100)


# In[10]:


plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.title('function')
plt.grid()
plt.scatter(xlist,dfx(xlist))
plt.plot(x_,cubic(x_))
plt.subplot(1,2,2)
plt.title('slope')
plt.grid()
plt.plot(x_,dfx(x_))


# In[8]:


plt.plot(x_,dfx(x_))


# In[9]:


newx=2
prev=0
cons=0.001
n=0
xlist=[]
while n!=500:
    prev=newx
    grad=dfx(prev)
    newx=prev-cons*grad
    xlist.append(newx)
    n+=1
print("minimum at ",newx)
print("slope of min ",dfx(newx))
print("value of function", cubic(newx))
xlist=np.array(xlist)


# # Example 2- Multiple minima vs initial guess and advanced function
# 
# # $$g(x)=x^4-4x^2+5$$
# 

# In[11]:


x_2 = np.linspace(start=-2,stop=2,num=1000)

def fun(x):
    return x**4-4*(x**2)+5

def dfx2(x):
    return 4*x**3-8*x


# In[12]:


plt.figure(figsize=[20,5])
plt.xlim(-2,2)
plt.ylim(0.5,5.5)
plt.subplot(1,2,1)
plt.title('function')
plt.grid()
plt.scatter(x_2,fun(x_2))
plt.plot(x_2,fun(x_2),linewidth=1)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)
plt.title('slope')
plt.grid()
plt.plot(x_2,dfx2(x_2))


# # gradient descent as python function when we have 2 local minima

# In[13]:


def gradient_descent(derivative_fun,initial_value,const=0.02,prec=0.001):
    newx=initial_value
    n=0
    xlist=[]
    while n!=500:
        prev=newx
        grad=dfx2(prev)
        newx=prev-const*grad
        step=abs(newx-prev)
        xlist.append(newx)
        if step<prec:
            break
        n+=1
    print("minimum at ",newx)
    print("slope of min ",derivative_fun(newx))
    print("value of function", fun(newx)) 
    return np.array(xlist),n
xlist,n=gradient_descent(dfx2,0)
print("after steps",n)


# In[14]:


plt.figure(figsize=[20,5])
plt.xlim(-2,2)
plt.ylim(0.5,5.5)
plt.subplot(1,2,1)
plt.title('function')
plt.grid()
plt.scatter(x_2,fun(x_2))
plt.plot(x_2,fun(x_2),linewidth=1)
plt.scatter(xlist,fun(xlist))
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)
plt.title('slope')
plt.grid()
plt.plot(x_2,dfx2(x_2))
plt.scatter(xlist,dfx2(xlist),color='red')


# # Example 3- Divergence,Overflow and python tuples
# $$h(x)=x^5-2*x^4+2$$

# In[15]:


x_3=np.linspace(start=-2.5,stop=2.5,num=1000)

def h(x):
    return x**5-2*x**4+2

def dh(x):
    return 5*x**4-8*x**3


# In[ ]:





# In[16]:


def gradient_descent(derivative_fun,initial_value,const=0.02,prec=0.001):
    newx=initial_value
    n=0
    xlist=[]
    while n!=500:
        prev=newx
        grad=derivative_fun(prev)
        newx=prev-const*grad
        step=abs(newx-prev)
        xlist.append(newx)
        if step<prec:
            break
        n+=1
    print("minimum at ",newx)
    print("slope of min ",derivative_fun(newx))
    print("value of function", h(newx)) 
    return np.array(xlist),n
xlist,n=gradient_descent(dh,0.2)
print("after steps",n)


# In[17]:


plt.figure(figsize=[20,5])
plt.subplot(1,2,1)
plt.xlim(-1.2,2.5)
plt.ylim(-1,4)
plt.title('function')
plt.grid()
plt.scatter(x_3,h(x_3))
plt.plot(x_3,h(x_3),linewidth=1)
plt.scatter(xlist,h(xlist))
plt.subplot(1,2,2)
plt.xlim(-1,2)
plt.ylim(-4,5)
plt.title('slope')
plt.grid()
plt.plot(x_3,dh(x_3))
plt.scatter(xlist,dh(xlist),color='red')


# In[ ]:




