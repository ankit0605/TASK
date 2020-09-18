
#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement  Unsupervised Learning
# 
# From the given ‘Iris’ dataset, predict the optimum number of
# clusters and represent it visually.
# 

# In[1]:


import pandas as pd  # to import dataset
import numpy as np   # to perform computational tasks
import plotly
import plotly.express as px
import matplotlib as plt
import plotly.offline as pyo 
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
import os


# In[2]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


iris=pd.read_csv('C:/Users/user/Desktop/IRIS.csv')


# In[4]:


iris


# In[5]:


iris.shape


# In[ ]:





# In[85]:


y=px.scatter(iris,x='species',y='petal_width',size='petal_width',color='petal_width')


# In[86]:


y


# In[84]:


px.scatter(iris,x='species',y='petal_length',size='petal_length',color='petal_length')


# In[ ]:





# In[9]:


px.bar(iris,x='species',y='petal_width')


# In[ ]:





# In[ ]:





# In[10]:


iris.iplot(kind='bar',x=['species'],y=['petal_width'])


# In[ ]:





# In[11]:


px.line(iris,x='species',y='petal_width')


# In[ ]:





# In[12]:


px.scatter_matrix(iris,color='species',title='Iris',dimensions=['sepal_length','sepal_width','petal_width','petal_length'])


# In[ ]:





# ## Data preprocessing

# In[ ]:





# In[13]:


iris


# In[14]:


X=iris.drop(['species'],axis=1)


# In[15]:


X


# In[16]:


y=iris['species']


# In[17]:


y


# In[18]:


#label Encoding


# In[19]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)


# In[20]:


y


# In[21]:


X


# In[22]:


X=np.array(X)


# In[23]:


X


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[26]:


X_test


# In[ ]:





# In[27]:


X_train.shape


# In[ ]:





# ## Decision Tree

# In[28]:


from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[29]:


prediction_dt=dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,prediction_dt)*100


# In[30]:


prediction_dt


# In[31]:


accuracy_dt


# In[ ]:





# ## For custom Prediction  in Decision Tree

# In[32]:


Category=["Iris-Setosa","Iris-Versicolor","Iris-Virginica"]


# In[33]:


X_dt=np.array([[5.7,3,4.2,1.2]])
X_dt_prediction=dt.predict(X_dt)


# In[34]:


X_dt_prediction


# In[35]:


X_dt_prediction[0]
print(Category[int(X_dt_prediction[0])])


# **Array 1 represents Iris-Versicolor**

# In[ ]:





# ## K-Nearest Neighbors

# In[ ]:





# In[36]:


# to better analyze the data 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[37]:


X_train_std


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_std,y_train)


# In[39]:


predict_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,predict_knn)


# In[40]:


accuracy_knn*100


# In[ ]:





# ## For Custom prediction in KNN

# In[ ]:





# In[41]:


Category=["Iris-Setosa","Iris-Versicolor","Iris-Virginica"]


# In[42]:


X_knn=np.array([[1,1,1,1]])
X_knn_prediction=knn.predict(X_knn)


# In[43]:


X_knn_std=sc.transform(X_knn)


# In[44]:


X_knn_prediction=dt.predict(X_knn_std)


# In[45]:


X_knn_prediction[0]
print(Category[int(X_knn_prediction[0])])


# **It represents Iris Setosa**

# In[ ]:





# ## To get the KNN Value

# In[46]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    knn_prediction=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,knn_prediction)
    scores_list.append(accuracy_score(y_test,knn_prediction))


# In[47]:


scores


# In[ ]:





# In[48]:


#with matplotlib
plt.plot(k_range,scores_list)


# In[ ]:





# In[49]:


#with plotly
px.line(x=k_range,y=scores_list)


# **With this we can easiy know what should be the accurate value to get the best results**

# In[ ]:





# ## K - Means Clustering

# In[ ]:





# In[50]:


y


# In[ ]:





# In[54]:


colormap=np.array(['Red','Green','Blue'])
fig=plt.scatter(iris['petal_length'],iris['petal_width'],c=colormap[y],s=50)


# In[ ]:





# To find the optimum no. of clusters:

# In[89]:


x = iris.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# **From the Elbow Method ,we know the optimum value is 3**

# In[90]:


km=KMeans(n_clusters=3,random_state=2,n_jobs=4)
km.fit(X)


# In[61]:


centers=km.cluster_centers_
print(centers)


# In[ ]:





# In[62]:


km.labels_


# In[ ]:





# In[63]:


Category=["Iris-Setosa","Iris-Versicolor","Iris-Virginica"]


# In[ ]:





# In[64]:


colormap=np.array(['Red','Green','Blue'])
fig=plt.scatter(iris['petal_length'],iris['petal_width'],c=colormap[km.labels_],s=50)


# In[ ]:





# In[83]:


new_labels=km.labels_
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(X[:,2],X[:,3],c=y,cmap='gist_rainbow',edgecolor='k',s=150)
axes[1].scatter(X[:,2],X[:,3],c=y,cmap='jet',edgecolor='k',s=150)
axes[0].set_title('Actual',fontsize=18)
axes[1].set_title('Predicted',fontsize=18)


# In[ ]:





# ## For Custom Prediction in K Means Clustering

# In[ ]:





# In[79]:


Category=["Iris-Setosa","Iris-Versicolor","Iris-Virginica"]


# In[81]:


X_kmeans=np.array([[1,1,1,1]])
X_kmeans_prediction=km.predict(X_kmeans)
X_kmeans_prediction[0]
print(Category[int(X_kmeans_prediction[0])])


# **It represents Iris Versicolor**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
