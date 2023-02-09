#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


data =pd.read_csv("D:/titanic.csv")


# In[4]:


data=data.dropna()


# In[5]:


data.head()


# In[6]:


y = data['Survived']
x = data[['Age','Fare']]


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.15)


# In[9]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[10]:


y_predict = model.predict(X_test)
accuracy_score(y_predict,y_test)


# In[11]:


a = np.array([[6,26.550]])
model.predict(a)


# In[ ]:





# In[ ]:




