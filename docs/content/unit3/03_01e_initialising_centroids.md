# Initialising the centroids

**You might have realised that changing the initial centroids has a drastic effect on how long the K-means algorithm takes to converge. Let us study this phenomena.**

In figure 1.7 below we see an example where the clusters are either being merged or correctly kept separated depending on the initial selection of centroids.

![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG179.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG180.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG190.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG192.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG191.png)

**<p style="text-align: center;">Figure 1.7:** *K-means not working well due an unlucky choice of the initial centroids. Images are reproduced from slides by Tan et al (2019), <a href="https://www-users.cs.umn.edu/~kumar001/dmbook/slides/chap7_basic_cluster_analysis.pdf" target="_blank">Introduction to Data Mining</a>, with kind permission of the authors.*</p>

<a href="https://www-users.cs.umn.edu/~kumar001/dmbook/slides/chap7_basic_cluster_analysis.pdf" target="_blank">Introduction to Data Mining</a>

Let us see another example where we have 10 clusters (10 centroids). As we can see below, the data is actually divided into two pairs of clusters. If we placed two centroids in one of the paired clusters then the K-means will be able to correctly adjust the centroids and reach a satisfactory result.

![Graph illustrating k-means working well due to good, informative choice of initial centroids.](images/DS_IMG181.png)
![Graph illustrating k-means working well due to good, informative choice of initial centroids.](images/DS_IMG182.png)
![Graph illustrating k-means working well due to good, informative choice of initial centroids.](images/DS_IMG183.png)
![Graph illustrating k-means working well due to good, informative choice of initial centroids.](images/DS_IMG184.png)

**<p style="text-align: center;">Figure 1.8:** *K-means working well due to a good informative choice of the initial centroids. Images are reproduced from slides by Tan et al (2019), <a href="https://www-users.cs.umn.edu/~kumar001/dmbook/slides/chap7_basic_cluster_analysis.pdf" target="_blank">Introduction to Data Mining</a>, with kind permission of the authors.*</p>

However, if we shift one of the initial centroids (second pair from the left) to another pair (last pair on the right), then the centroids end up mingled for these two pairs of clusters (as we can see in on the right), then the centroids end up mingled for these two pairs of clusters (as we can see in iteration 4 below).

![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG185.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG186.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG187.png)
![Graph illustrating k-means not working well due to an unlucky choice of initial centroids.](images/DS_IMG188.png)

**<p style="text-align: center;">Figure 1.9:** *K-means not working well due an unlucky choice of the initial centroids. Images are reproduced from slides by Tan et al (2019), <a href="https://www-users.cs.umn.edu/~kumar001/dmbook/slides/chap7_basic_cluster_analysis.pdf" target="_blank">Introduction to Data Mining</a>, with kind permission of the authors.*</p>

Hence it is paramount to come up with a viable strategy to initialise the centroids.

##Hill-climbing centroids

**One strategy to overcome this issue is by randomising the initial centroids and performing multiple runs of the K-means and then selecting the one that produce the least SSE.**

This called hill-climbing algorithm in AI and you might come across it in the Algorithms module. In all cases, the idea is simple. Just perform the same algorithm multiple times, each time with a randomly selected initial centroid, and then select the clusters from the run that performs the best in terms of SSE. The downside of this strategy is that it can be costly especially when we consider large datasets in which case it becomes infeasible, but it can work well with small to medium datasets. You saw an example of this strategy in the exercise that you performed at the end of section 1d of this lesson.

##Furthest apart centroids: K means++

**Another strategy that might be less expensive is to make a deliberate choice of the centroid.**

Rather than the random choice we saw above, we select the centroids one by one gradually, in a way to maximise the separation between them (i.e. to be as far from each other as possible). This strategy might seem risky if we consider outliers, because by definition they are furthest from other data points in the dataset. However, it turns out that this strategy works well in practice since the probability of a centroid landing on an outlier in a large dataset is quite small. This is the idea of K-means++ algorithm which we will show in the next section after we talk more about another strategy to deal with large datasets. However, bear in mind that we still want the randomness of picking a set of data points as initial centroids. Therefore, we will assign a probability of picking a data point as an initial centroid based on its distance to the nearest centroid $\boldsymbol{c}^{\prime}$. This implies that we are comparing current data point $x$ with the set of all already picked centroids $\boldsymbol{c}_{i}$ where $i$ is still not reached $K$ yet. The probability is given as:

$$
p\left(\mathbf{x}_{n}\right)=\frac{d\left(\mathbf{x}_{n}, \boldsymbol{c}^{\prime}\right)^{2}}{\sum_{x \in X} d\left(\mathbf{x}, \boldsymbol{c}^{\prime}\right)^{2}}
$$

So the **further** the data point $\mathbf{x}_{n}$ is to the **nearest centroid**, the more likely it will be picked as the next centroid. We repeat the process until we pick all $K$ centroids.
