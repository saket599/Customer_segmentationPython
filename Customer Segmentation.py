#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Mall_Customers.csv")


# In[3]:


df


# In[4]:


df.describe().T


# # Univariate Analysis

# In[5]:


sns.distplot(df["Annual Income (k$)"]);


# In[6]:


df.columns


# In[7]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i]);


# In[8]:


sns.kdeplot(df["Annual Income (k$)"],shade=True,hue=df["Gender"]);


# In[9]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(df[i],shade=True,hue=df["Gender"]);


# In[10]:


df["Gender"].value_counts(normalize=True)


# # Bivariate Analysis

# In[11]:


sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[12]:


sns.pairplot(df,hue='Gender')


# In[13]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[14]:


df.corr()


# In[15]:


sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[16]:


clustering1 = KMeans(n_clusters=3)


# In[17]:


clustering1.fit(df[['Annual Income (k$)']])


# In[18]:


clustering1.labels_


# In[19]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[20]:


df['Income Cluster'].value_counts()


# In[21]:


clustering1.inertia_


# In[22]:


inertia_scores = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[23]:


inertia_scores


# In[24]:


plt.plot(range(1,11),inertia_scores)


# In[25]:


df.columns


# In[26]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Bivariate Clustering

# In[27]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[28]:


inertia_scores2 = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)


# In[29]:


plt.plot(range(1,11),inertia_scores2)


# In[30]:


centre =pd.DataFrame(clustering2.cluster_centers_)
centre.columns = ['x','y'] 
centre


# In[34]:


plt.figure(figsize=(8,10))
sns.scatterplot(data=centre, x = centre['x'], y = centre['y'], s=100,marker = '*')
sns.scatterplot(data=df, x ='Annual Income (k$)', y='Spending Score (1-100)', hue= 'Spending and Income Cluster', palette='tab10')


# In[32]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[33]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Mutlivariate Clustering

# In[37]:


from sklearn.preprocessing import StandardScaler 


# In[39]:


scale = StandardScaler()


# In[42]:


dff = pd.get_dummies(df,drop_first=True)


# In[43]:


dff.head()


# In[46]:


dff.columns


# In[48]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff


# In[49]:


dff = scale.fit_transform(dff)


# In[50]:


dff = pd.DataFrame(scale.fit_transform(dff))


# In[52]:


inertia_scores3 = []
for i in range(1,11):
    kmeans3 = KMeans(n_clusters=i)
    kmeans3.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_scores3)


# In[ ]:




