#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[87]:


data1=pd.read_csv("adult.csv")
data1


# In[54]:


data=pd.read_csv("adult.csv")


# In[55]:


data


# In[56]:


from sklearn.preprocessing import LabelEncoder


# In[57]:


lb=LabelEncoder()


# In[58]:


data['Workclass']=lb.fit_transform(data['Workclass'])


# In[59]:


data['Marital Status']=lb.fit_transform(data['Marital Status'])


# In[60]:


data['Occupation']=lb.fit_transform(data['Occupation'])


# In[61]:


data['Relationship']=lb.fit_transform(data['Relationship'])


# In[ ]:





# In[62]:


data['Native Country'].value_counts()


# In[63]:


data['Native Country']=lb.fit_transform(data['Native Country'])


# In[64]:


data['Race']=lb.fit_transform(data['Race'])


# In[65]:


data


# In[66]:


data['Gender']=data['Gender'].apply(lambda x:1 if x=='Male'  else 1)


# In[67]:


data['Income']=data['Income'].apply(lambda x:0 if x=='<=50K' else 1)


# In[68]:


data['Education']=lb.fit_transform(data['Education'])


# In[69]:


data


# In[70]:


data.isnull().sum()


# In[71]:


from sklearn.ensemble import RandomForestClassifier


# In[72]:


rf=RandomForestClassifier(n_estimators=100,bootstrap=True)


# In[73]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[74]:


x_train,x_test,y_train,y_test=train_test_split(data.drop(columns='Income'),data['Income'],test_size=0.2,random_state=2)


# In[ ]:





# In[75]:


rf.fit(x_train,y_train)


# In[76]:


y_pred=rf.predict(x_test)


# In[77]:


y_pred


# In[78]:


from sklearn.metrics import accuracy_score


# In[79]:


accuracy_score(y_test,y_pred)


# SVM alogorithm 

# In[80]:


from sklearn import svm


# In[81]:


regr = svm.SVR()
regr.fit(x_train, y_train)


# In[82]:


ypred1=regr.predict(x_test)


# In[83]:


accuracy_score(y_test,ypred1)


# crossvalscore

# In[103]:


cvs=cross_val_score(rf,x_train,y_train,cv=10)
print(cvs)


# In[ ]:





# In[ ]:





# In[ ]:




