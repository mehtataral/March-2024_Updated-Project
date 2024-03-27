# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:47:49 2023

@author: Livewire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Reading the excel file
df=pd.read_csv(r"E:\Taral\DataScience\Project\Mall_Customer.csv")
df
df.shape
df.size

df.head()
df.tail()

df.columns
df
df.dtypes.value_counts()

df.info()

df["  Spending Score (1-100)   "]  = df["Spending Score (1-100)"] 
df["Spending Score (1-100)"] 
df["  Spending Score (1-100)   "]

df.drop("Spending Score (1-100)",axis ="columns",inplace =True)
df
df.columns  = df.columns.str.replace(" ","")
df["SpendingScore(1-100)"]

df.drop(columns ="CustomerID",inplace =True)
df
df.describe()

#---------------------------------------------------------------------

sns.boxplot(df)


ax = sns.countplot(data =df,x  = "Genre")
for x in ax.containers:
    ax.bar_label(x)


df["AnnualIncome(k$)"].plot(kind ="kde")

sns.scatterplot(data =df, x ="AnnualIncome(k$)",y  ="SpendingScore(1-100)",hue ="Genre")

# We can see that most of the customer data points lies at annual income(40-70) and spending score (40-60).

sns.scatterplot(data =df,x = "Age",y = "AnnualIncome(k$)",hue ="Genre")
sns.scatterplot(data =df,x = "Age",y = "SpendingScore(1-100)",hue ="Genre")

# from age 40 to 60 score is 20 to 60
# from age 20 to 40 score is 40 to 100
# from age 60 to 70 score is 40 to 60



df.Genre = np.where(df.Genre =="Male",1,0)
df.Genre
from sklearn.preprocessing import StandardScaler
# Scaling desired columns for modelling
scaler=StandardScaler()
scaled_val=scaler.fit_transform(df[["AnnualIncome(k$)","SpendingScore(1-100)"]])


scaled_val
features=pd.DataFrame(scaled_val,columns=df.columns[2:4].tolist())
features



# To get optimal number of clusters (K) for KMeans cluster algo..
from sklearn.cluster import KMeans

nc=[]
sse=[]
for i in range(1,11):
    km = KMeans(i)
    km.fit(features)
    nc.append(i)
    sse.append(km.inertia_)
plt.figure(figsize=(10,6))
plt.plot(nc,sse,'rs:')

# :We interpret that, i took the values for k is (1-20) &
# the line is steadily decreasing as k values increase.
# so considering elbow at k=5 clusters


kval=5

kmeans=KMeans(n_clusters=kval,max_iter=10,random_state=69)
kmeans.fit(features)
labels=kmeans.labels_
labels
labels.size
labels.shape

centroids=kmeans.cluster_centers_
centroids

e=kmeans.inertia_
e

itr=kmeans.n_iter_
itr
new_features=features.assign(clusters=pd.DataFrame(labels))
new_features
    
plt.figure(figsize=(10, 6))

# Plot each cluster's data points with different colors
clr=["red","green","blue","orange","purple"]
for cluster_num in range(kval):
    plt.scatter(x=new_features[new_features.clusters == cluster_num]['AnnualIncome(k$)'],
               y= new_features[new_features.clusters == cluster_num]['SpendingScore(1-100)'],marker='o'
               ,c=clr[cluster_num],label=f'Cluster {cluster_num}')

# Plot the cluster centers
plt.scatter(x=centroids[:, 0], y=centroids[:, 1],
            c='black', marker='X', s=150, label='Centroids')

plt.xlabel('AnnualIncome')
plt.ylabel('SpendingScore')
plt.title('K-means Clustering')
plt.legend()
plt.show()



from sklearn import metrics
#Evaluation Metric
score=metrics.silhouette_score(features,kmeans.labels_)
print("Silhouette_Score Coefficient : {:.2f}".format(score))

# Silhouette_Score ranges from -1 to 1,which near to one is best and nearr to -1 is worst. Since we got coffecient as 0.55</i> in which datapoints are very Moderately compact with the clusters.



# Conclusion :
# From the entire clustering analysis,it is to be interpreted that,

# Cluster 0 (Red):
# The data points in this cluster represents that the customers with
# an average annual income tends to have average spending
# score(-0.5 to 0.5)..So,these customers are the very balanced ones.

# Cluster 1 (Green):
# The data points in this cluster represents that the customers with 
# higher annual income tends to have higher spending
# score between (-0.5 to -2).So,these customers are treated as 
# target and more profitable ones to the business domain.

# Cluster 2 (Blue):
# The data points in this cluster represents that the customers with
# an higher annual income tends to have lower spending
# score(-2 to -0.5).So,we can treat these customers are well
# planned and careful ones.

# Cluster 3 (Orange):
# The data points in this cluster represents that the customers with 
# lower annual income tends to have higher spending score(0.4 to 2).
# So,these customers are the are balanced ones or very careless ones.

# Cluster 4 (Purple):
# The data points in this cluster represents that the customers with
# lower annual income tends to have low spending score(-0.5 to -2).
# So,these customers are the very balanced and careful ones.



# https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Deployment
# https://www.kaggle.com/code/bhanux18/mall-customer-clustering















