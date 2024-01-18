#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Understanding the data
import pandas as pd # analysis
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns #data visualization

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans


# In[3]:


df = pd.read_csv("C:\\Users\\harsh\\Documents\\Python stuff\\mall-customers-data.csv")


# In[4]:


display(df.head())


# In[5]:


df.head(20)


# In[6]:


df.tail(20)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.corr(numeric_only=True)


# In[11]:


df


# In[12]:


sns.set(style="whitegrid")
sns.displot(df["annual_income"], kde=True, color="dodgerblue")
 
plt.title("Distribution of annual income(in k$)")
plt.xlabel("Range of annual income(in k$)")
plt.ylabel("Counts")

plt.show()


# In[13]:


# most annual income is in range are 60 and 80 (k$)


# In[14]:


# distribution of age
sns.set(style="whitegrid")
sns.displot(df["age"], kde = True, color="dodgerblue")

plt.title("Distribution of Age", fontsize=20)
plt.xlabel("Range of Age")
plt.ylabel("Count")
plt.show()


# In[15]:


# most buyers are aged between 30-35


# In[16]:


df


# In[17]:


# distribution of spending score
sns.set(style="whitegrid")
sns.displot(df["spending_score"], kde=True, color="dodgerblue", binwidth=10)

plt.title("Distribution of spending score")
plt.xlabel("Range of spending score")
plt.ylabel("Counts")

plt.show


# In[18]:


# highest spending score is between 40-50


# In[19]:


# gender analysis
genders = df["gender"].value_counts()

colors = {
    "Male": "dodgerblue",
    "Female": "orange"
}

sns.set(style="whitegrid")
sns.barplot(x=genders.index, y=genders.values, palette=colors)

plt.show()


# In[20]:


genders.index


# In[21]:


df


# In[22]:


X = df[["annual_income", "spending_score"]]
X.head(20)


# In[23]:


# scatter plot of above data
sns.scatterplot(x= "annual_income", y="spending_score", data=X, s=60, color="dodgerblue" )

plt.title("Spending score vs annual income")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.show()


# In[24]:


#wcss - Within-Cluster Sum of Squares
#sum(sq(euclidean distance between centroid and data point)


# In[25]:


pip install --upgrade threadpoolctl


# In[ ]:





# In[33]:


wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i, n_init="auto")
    k_means.fit(X)
    wcss.append(k_means.inertia_)
    
# Elbow Curve
plt.plot(range(1,11), wcss)
plt.plot(range(1,11), wcss, linewidth=2.5, color="red", marker = "8")

plt.title("Elbow Curve")
plt.xlabel("Number of clusters(values of k)")
plt.ylabel("WCSS values")
plt.xticks(np.arange(1, 11,1))
plt.show()


# In[34]:


# Training model using Unsupervised leaning algorithm(K-Means)


# In[38]:


kmeansmodel = KMeans(n_clusters = 5, init='k-means++', random_state=0)


# In[39]:


y_kmeans = kmeansmodel.fit_predict(X)


# In[46]:


kmeans = KMeans(n_clusters=5)  # Assuming 5 clusters
kmeans.fit(X)


# In[52]:


plt.scatter(X.iloc[y_kmeans == 0,0], X.iloc[y_kmeans == 0, 1], s=80, c = "red", label ='Customer1')
plt.scatter(X.iloc[y_kmeans == 1,0], X.iloc[y_kmeans == 1, 1], s=80, c = "blue", label ='Customer2')
plt.scatter(X.iloc[y_kmeans == 2,0], X.iloc[y_kmeans == 2, 1], s=80, c = "black", label ='Customer3')
plt.scatter(X.iloc[y_kmeans == 3,0], X.iloc[y_kmeans == 3, 1], s=80, c = "cyan", label ='Customer4')
plt.scatter(X.iloc[y_kmeans == 4,0], X.iloc[y_kmeans == 4, 1], s=80, c = "pink", label ='Customer5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='magenta', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[ ]:




