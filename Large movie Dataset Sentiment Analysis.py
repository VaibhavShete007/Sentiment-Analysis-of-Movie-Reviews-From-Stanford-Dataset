#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy import *
import matplotlib.pyplot as plt


# In[2]:


X_data = loadtxt("Movie Sentiment analysis/X_train.txt") 
print (X_data.shape)
X = X_data


# In[3]:


y_data = loadtxt("Movie Sentiment analysis/y_train.txt", dtype = int) 
print (y_data.shape)
y = y_data


# In[4]:


from sklearn.utils import shuffle
X_new, y_new = shuffle(X, y)

X_train = X_new[:1000]
y_train = y_new[:1000]
X_test = X_new[1000:]
y_test = y_new[1000:]


# In[5]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[6]:


#Training:
logreg.fit(X_train,y_train)
ypredtrain=logreg.predict(X_train)
print("Accuracy on Training Set:",logreg.score(X_train,y_train)*100)
#accuracy on traing set in %!!


# In[7]:


#Testing:
ypredtest=logreg.predict(X_test)
print("Accuracy on Test Set:",logreg.score(X_test,y_test)*100)
#Accuracy on test set in %!!


# In[11]:


ipython nbconvert _ to script abc.ipynb


# In[12]:


jupyter nbconvert --to script *.ipynb


# In[ ]:




