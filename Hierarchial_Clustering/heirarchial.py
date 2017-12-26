# **********************************************************************************************************
# Finding the cluster of clients based on annual income and spending score using Hierarchial Clustering
# **********************************************************************************************************

# Importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

# Finding the optimal number of clusters using dendograms
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance btw the centroids of the clusters')
plt.show()
# From the above dendogram, we can see that the optimal number of clusters is 5

# Fitting the hierarchial to the datset
from sklearn.cluster import AgglomerativeClustering
hcCluster= AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean', linkage = 'ward')
y_hc = hcCluster.fit_predict(X)

# plotting the graph
plt.scatter(X[y_hc == 0,0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster5')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()