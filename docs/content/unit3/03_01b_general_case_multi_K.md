# General case multi clusters: K-means algorithm

**So far, you might have realised that the centres depend on the clusters, and at the same time the clusters are identified through the centres.**

This algorithm has a chicken and egg dilemma, and in data mining and machine learning once we are faced with such scenarios we reside into iteratively improving one and freezing the other and then vice versa. In other words, we will iteratively: 1 – change our cluster and calculate their centres, and then 2 –  re-assign the data points into clusters and then recalculate the centres. We do that iteratively until the algorithms converge i.e. the clusters stop changing (or change very little).

What we have talked about so far was actually the **K-means** algorithm. The name comes from assuming in priori that we have **K clusters** and from the fact that we take the **means** of the clusters (centroids), and based on them we reassign the data points and then calculate the means again and so on. The algorithm is shown below:

!!! algorithm-heading "Algorithm 1: Basic K-means"

    **Input:**

    Dataset $X$=$\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{\mathrm{N}}\right\}$

    $K$: The number of clusters

    **Output**: Cluster labels $\boldsymbol{l}=\left[l_{1}, l_{2}, \ldots, l_{N}\right] \quad l_{n} \in\{1, \ldots, K\}$

    **K-means** (**X**, *K*):

    !!! algorithm ""
        $\boldsymbol{C} \boldsymbol{t}=\left\{\mathbf{c}_{i}\right\}$ Select $K$ points from $\mathbf{X}$ as initial centroids, randomly

        **Repeat**

        !!! algorithm ""
            **For** each $\mathbf{x}_{n}$ in $\mathbf{X}$  

            !!! algorithm ""

                $l_{n}=\arg \min _{i} d\left(\mathbf{c}_{i}, \mathbf{x}_{n}\right) \quad i=1, \ldots, K$ <span style="float: right;"># Form K Clusters by assigning each point to its closest centroid</span>


            **For** each Cluster $C_{i}$ of size $\left|\boldsymbol{C}_{i}\right|$  

            !!! algorithm ""    

                $\boldsymbol{c}_{\boldsymbol{i}}=\frac{1}{\left|\boldsymbol{C}_{i}\right|} \sum_{\mathbf{x} \in \boldsymbol{C}_{i}} \mathbf{x}$ <span style="float: right;"># Recalculate the centroids for each cluster </span>


            **until** the centroids do not change

        **Return** the labels $\boldsymbol{l}=\left[l_{1}, l_{2}, \ldots, l_{N}\right]$
