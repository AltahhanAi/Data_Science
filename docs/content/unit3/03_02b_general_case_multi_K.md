# General case multi clusters: K-means algorithm

So far, you might have realised that the centres depend on the clusters, and at the same time the clusters are identified through the centres. This algorithm has a chicken and egg dilemma, and in data mining and machine learning once we are faced with such scenarios we reside into iteratively improving one and freezing the other and then vice versa. In other words, we will iteratively: 1 – change our cluster and calculate their centres, and then 2 –  re-assign the data points into clusters and then recalculate the centres. We do that iteratively until the algorithms converges i.e. the clusters stop changing (or change very little).

What we have talked about so far was actually the **k-means** algorithm. The name comes from assuming in priori that we have **k clusters** and from the fact that we take the **means** of the clusters (centroids), and based on them we reassign the data points and then calculate the means again and so on. The algorithm is shown below:

!!! algorithm-heading "Algorithms 1: Basic K-means"

Algorithms 1: Basic K-means
Input:
Dataset X={x_1,x_2,…,x_N}
K: The number of clusters.
Output: Cluster Labels C=[l_1,l_2,…,l_N ]     l_n∈{1,…,K}
K-means(X,K):
Cent={c_i} Select K points form X as initial centroids or at random
Repeat
For each x_n  in   X
l_n=〖arg min┬i〗⁡〖d(c_i,x_n)〗    i=1,…,K 	  # Form K Clusters by assigning each point to its closest centroid

For each Cluster C_i of size m_i
c_i=1/m_i  ∑_(x∈C_i)▒x 	  # Recalculate the centroids for each cluster

until the centroids do not change
return the labels C=[l_1,l_2,…,l_N ]
