# -*- coding: utf-8 -*-
"""
@author: arjun
"""
# **********************************************************************************************************
# Finding the cluster of clients based on annual income and spending score
# **********************************************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Hence, we can see that 5 is the optimal number of clusters
# Applying K-Means to the dataset with K=5
kmeans_optimal = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans_optimal.fit_predict(X)

# plotting the graph
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster5')

# Plotting the centroid
plt.scatter(kmeans_optimal.cluster_centers_[:,0], kmeans_optimal.cluster_centers_[:,1], s= 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()