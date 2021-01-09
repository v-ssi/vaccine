#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression


# In[5]:


data = pd.read_csv("data.csv")
print(data.head())


# In[6]:


plt.figure(figsize = (8,8))
sb.pairplot(data)


# In[7]:


sb.heatmap(data.corr())


# In[8]:


X = data.drop(['Usable','Traffic Density'],axis=1)
Y = data['Usable']
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2,random_state=4)


# In[22]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[23]:


Y_pred = lr.predict(X_test)
print(Y_pred)
print(Y_test)


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score
accuracy_score(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))
precision_score(Y_test, Y_pred, average=None, zero_division=1)
print(precision_score)
confusion_matrix(Y_test, Y_pred)

