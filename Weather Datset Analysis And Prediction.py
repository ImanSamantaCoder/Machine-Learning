#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[97]:


df=pd.read_csv(r"D:\ALL DATASET\weathernew.csv")


# In[98]:


new_df=df


# In[99]:


#new_df['Date']=df['Formatted Date'].apply(lambda x:x.split()[0])
new_df['DATE AND TIME']=df['Formatted Date'].apply(lambda x:x.split()[0])+" "+df['Formatted Date'].apply(lambda x:x.split()[1].split(".")[0])
new_df['DATE AND TIME']=pd.to_datetime(new_df['DATE AND TIME'])


# In[100]:


new_df=new_df.drop(columns=['Formatted Date','Loud Cover','Daily Summary'])


# In[101]:


new_df.info()


# In[102]:


new_df.shape


# In[103]:


new_df['Summary'].value_counts()


# In[104]:


new_df= new_df[(new_df["Summary"] == "Clear") | (new_df["Summary"] == "Overcast")|(new_df["Summary"] == "Foggy")]


# In[106]:


#new_df['Summary']=new_df['Summary'].replace({'Partly Cloudy':4})
new_df['Summary'].value_counts()


# In[11]:


new_df['Summary'].value_counts()


# In[12]:


new_df.isnull().sum()


# In[13]:


new_df.dropna(inplace=True)


# In[14]:


new_df


# In[15]:


new_df.isnull().sum()


# In[16]:


new_df['Year']=new_df['DATE AND TIME'].dt.year
new_df['Month']=new_df['DATE AND TIME'].dt.month_name()
new_df['Day']=new_df['DATE AND TIME'].dt.day
new_df['day_name']=new_df['DATE AND TIME'].dt.day_name()
new_df['iso_week_number'] = new_df['DATE AND TIME'].dt.isocalendar().week
new_df['Hour']=new_df['DATE AND TIME'].dt.hour


# In[17]:


new_df


# In[18]:


new_df=new_df.drop(columns=['DATE AND TIME','day_name'])


# In[19]:


new_df


# In[20]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
new_df['Month']=lb.fit_transform(new_df['Month'])
new_df.info()


# In[21]:


new_df['Summary'].value_counts()


# In[22]:


new_df['Summary']=new_df['Summary'].replace({'Clear':2,'Overcast':1,'Foggy':3})


# In[23]:


new_df['Summary'].value_counts()


# In[24]:


new_df.info()


# In[25]:


new_df['Precip Type']=lb.fit_transform(new_df['Precip Type'])
new_df.info()


# In[ ]:





# In[26]:


new_df.info()


# In[27]:


new_df.sample(5)


# In[60]:


plt.boxplot(new_df['Temperature (C)'])


# In[61]:


percentile25=new_df['Temperature (C)'].quantile(.25)
percentile75=new_df['Temperature (C)'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Temperature (C)']=np.where(
new_df['Temperature (C)']>upper_limit,
upper_limit,
np.where(
new_df['Temperature (C)']<lower_limit,
lower_limit,
new_df['Temperature (C)']))


# In[62]:


#outlier removed finally
plt.boxplot(new_df['Temperature (C)'])


# In[63]:


plt.boxplot(new_df['Humidity'])


# In[64]:


percentile25=new_df['Humidity'].quantile(.25)
percentile75=new_df['Humidity'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Humidity']=np.where(
new_df['Humidity']>upper_limit,
upper_limit,
np.where(
new_df['Humidity']<lower_limit,
lower_limit,
new_df['Humidity']))


# In[65]:


plt.boxplot(new_df['Humidity'])


# In[66]:


plt.boxplot(new_df['Apparent Temperature (C)'])


# In[67]:


percentile25=new_df['Apparent Temperature (C)'].quantile(.25)
percentile75=new_df['Apparent Temperature (C)'].quantile(.75)
iqr=percentile75-percentile25
upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr
new_df['Apparent Temperature (C)']=np.where(
new_df['Apparent Temperature (C)']>upper_limit,
upper_limit,
np.where(
new_df['Apparent Temperature (C)']<lower_limit,
lower_limit,
new_df['Apparent Temperature (C)']))


# In[68]:


#outlier removed finally
plt.boxplot(new_df['Apparent Temperature (C)'])


# In[41]:


plt.boxplot(new_df['Wind Bearing (degrees)'])


# In[42]:


plt.boxplot(new_df['Visibility (km)'])


# In[43]:


plt.scatter(new_df['Temperature (C)'],new_df['Humidity'])
plt.xlabel("Temperature")
plt.ylabel("humidity")


# In[44]:


plt.figure(figsize=(500,400))
cat_plot=sns.catplot(data=new_df,x='Month',y='Temperature (C)',kind='bar',col='Year',col_wrap=3,sharex=False)
cat_plot.set_xticklabels(rotation=90)
plt.show()
sns.displot(data=new_df,x='Temperature (C)',kind='kde',col='Year',fill=True,col_wrap=3)
cat_plot.set_xticklabels(rotation=90)
plt.show()


# In[45]:


plt.figure(figsize=(50,40))
cat_plot=sns.catplot(data=new_df,x='Month',y='Humidity',kind='bar',col='Year',col_wrap=3,sharex=False)
cat_plot.set_xticklabels(rotation=90)
plt.show()


# In[46]:


plt.figure(figsize=(15,18))
ax=plt.subplot(projection='3d')
ax.scatter3D(new_df['Temperature (C)'],new_df['Humidity'],new_df['Visibility (km)'])
ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.set_zlabel('Visitbility')


# In[ ]:





# In[ ]:





# In[49]:


new_df


# In[ ]:





# In[50]:


new_df['Summary'].value_counts()


# In[69]:


sns.heatmap(new_df.corr(),annot=True, cmap='coolwarm', linewidths=.5)


# In[52]:


new_df=new_df.drop(columns=['Day','Hour','iso_week_number'])


# In[53]:


new_df


# In[54]:


x=new_df.iloc[:,1:]
y=new_df.iloc[:,0]


# In[55]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[86]:


# importing random forest classifier from assemble module 
from sklearn.ensemble import RandomForestClassifier 
# Create a Random forest Classifier 
clf = RandomForestClassifier(n_estimators = 100) 

# Train the model using the training sets 
clf.fit(x_train, y_train)


# In[87]:


y_pred1=clf.predict(x_test)


# In[88]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
accuracy_score(y_test,y_pred1)


# In[89]:


new_df['Summary'].value_counts()


# In[90]:


cm=confusion_matrix(y_test,y_pred1)
sns.heatmap(cm,annot=True,fmt='g',xticklabels=['Overcast','Clear','foggy'],yticklabels=['Overcast','Clear','foggy'])


# In[93]:


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(clf,x,y,cv=5,scoring='accuracy'))


# In[94]:


precision_score(y_test,y_pred1,average='weighted')


# In[95]:


recall_score(y_test,y_pred1,average='weighted')


# In[123]:


from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  


# In[124]:


y_pred= classifier.predict(x_test)  


# In[125]:


accuracy_score(y_test,y_pred1)


# In[126]:


np.mean(cross_val_score(classifier,x,y,cv=5,scoring='accuracy'))


# In[ ]:




