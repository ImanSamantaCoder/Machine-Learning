#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


data=pd.read_csv(r"D:\ALL DATASET\Iris.csv")


# In[6]:


data


# In[10]:


h=sns.PairGrid(data=data.iloc[:,1:],hue='Species')
h.map_diag(sns.violinplot)
h.map_offdiag(sns.scatterplot)


# In[13]:


sns.clustermap(data.iloc[:,[1,2,3,4]])


# In[35]:


fig,ax=plt.subplots(figsize=(28,18),nrows=2,ncols=2,)
ax[0][0].boxplot(data['SepalLengthCm'])
ax[0][0].set_title("SepalLength")
ax[0][1].boxplot(data['PetalWidthCm'])
ax[0][1].set_title("PetalWidthCm")
ax[1][0].boxplot(data['SepalWidthCm'])
ax[1][0].set_title("SepalWidthCm")
ax[1][1].boxplot(data['PetalLengthCm'])
ax[1][1].set_title("PetalLength")


# In[38]:


percentile25=data['SepalWidthCm'].quantile(.25)
percentile75=data['SepalWidthCm'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
data['SepalWidthCm']=np.where(
data['SepalWidthCm']>upper_limit,
upper_limit,
np.where(
data['SepalWidthCm']<lower_limit,
lower_limit,
data['SepalWidthCm']))


# In[40]:


sns.boxplot(data['SepalWidthCm'])


# In[42]:


data.sample(6)


# In[43]:


data=data.replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})


# In[44]:


data.sample(5)


# In[46]:


x=data.iloc[:,1:5]
y=data.iloc[:,5]


# In[49]:


x


# In[50]:


y


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[74]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
ds=DecisionTreeClassifier()
ds.fit(x_train,y_train)


# In[75]:


y_pred1=ds.predict(x_test)


# In[76]:


tree.plot_tree(ds)


# In[61]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)


# In[66]:


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(ds,x,y,cv=5,scoring='accuracy'))


# In[67]:


from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  


# In[68]:


y_pred= classifier.predict(x_test)  


# In[69]:


accuracy_score(y_test,y_pred)


# In[71]:


np.mean(cross_val_score(classifier,x,y,cv=5,scoring='accuracy'))


# In[ ]:




