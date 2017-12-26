# **********************************************************************************************************
# Finding the cluster of clients based on annual income and spending score using K-Means
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

