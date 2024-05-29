#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Importing all required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.tree import DecisionTreeClassifier


# In[25]:


univdata = pd.read_csv("C:/Users/Admin/Desktop/Myprojectdata.csv", delimiter = ",",encoding='unicode_escape' )
univdata

trimColNames = [name.strip() for name in univdata.columns]
univdata.columns = trimColNames


# In[ ]:


univdata['UnivName'] = univdata['UnivName'].astype(str)
print(univdata.dtypes)
univdata.shape
univdata.describe()

univdata = pd.get_dummies(univdata)
labels = np.array(univdata['UK Rank'])
univdata_list = list(univdata.columns)

univdata = np.array(univdata)


# In[174]:


#Finding the Correlation Matrix between the predicted and the output Variable
correlation_matrix = univdata.iloc[:,:].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
# Year, UK Rank and World class rank have a strong relation and hence can help in finding best university


# In[175]:


# Determining the Public and private universities as per UK Ranking
sns.catplot(data = univdata, x = "Type", y = "UK Rank")
# Determining the Public and private universities as per UK Ranking
sns.catplot(data = univdata, x = "Type", y = "World Class Rank")
#Result - There are less Private universities which has good world ranking

sns.displot(univdata['Avg fees UG (in pounds)'])


# In[210]:


X = univdata[["UK Rank", "World Class Rank"]]
# Visualize data point
plt.scatter(X["UK Rank"], X["World Class Rank"], c="blue")
plt.xlabel("UK Rank")
plt.ylabel("World Class Rank")
plt.show()


# In[211]:


K=3

# select random observation as a centriod 
Centroids = (X.sample(n=K))
plt.scatter(X["UK Rank"], X["World Class Rank"], c="blue")
plt.scatter(Centroids["UK Rank"], Centroids["World Class Rank"], c="red")
plt.xlabel("UK Rank")
plt.ylabel("World Class Rank")
plt.show()


# In[213]:


Centroids


# In[26]:


X = univdata["UK Rank"]
Y = univdata["World Class Rank"]
plt.plot(X, Y)
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
plt.title('Line Graph')
plt.show()


# In[27]:


x = univdata.iloc[:,4:6] # 1t for rows and second for columns
x


# In[28]:


identified_clusters = kMeans.fit_predict(x)
identified_clusters


# In[29]:


from sklearn.cluster import KMeans
kMeans = KMeans(3)
kMeans.fit(identified_clusters)


# In[244]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['UK Rank'],data_with_clusters['World Class Rank'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[207]:


#Training the Model

train_univdata, test_univdata,train_labels, test_labels = train_test_split(univdata, labels, test_size = 0.30, random_state = 1899)
print('Training Univdata Shape:', a_train.shape)
print('Testing Univdata Shape:', b_train.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[ ]:




