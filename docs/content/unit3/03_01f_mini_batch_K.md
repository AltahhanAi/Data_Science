# Mini-batch K-means++

**Since we have a loop that goes through *all instances/records* of the dataset then potentially we can employ the idea of mini-batch on clustering.**

A mini-batch is a compromise that aims at preserving the autonomy of the algorithm but also makes sure it is practical. Instead of iterating through the whole dataset before calculating the centroids, we can iterate through a part of the dataset to quickly update the centroids. This update is based on a sample of the dataset called a mini-batch. But then we repeat picking another mini-batch and update until either we covered all the dataset or repetitively until convergence. To cover the whole dataset we need to iterate through several mini-batches, this is called an epoch. The procedure is as follows:

1. We partition the dataset into several mini-batches. Each mini-batch involves several data points and all mini-batches have equal size (there might be a smaller remainder final batch that will be treated similarly). For example, if we have 100 data points in our dataset and we set the mini-batch size into 25, then we need 4 mini-batches to digest the whole dataset and each epoch will involve 4 mini-batches that constitute 4 iterations.

2. In each iteration, we update the clusters and recalculate the centroids based on only the data points inside the mini-batch $b_{i}$

3. In order to mitigate for the bias and variance that might occur due to a particular lucky (or unlucky) batch, we often shuffle the dataset, or we partition the dataset and take one mini-batch after the other.

The size of the mini-batch is a hyper parameter that should make a good compromise between iterating through all data points and taking just one data point.

Below, we show the K-means mini-batch algorithm which saves computation and makes more efficient use of the available data. Note that there will be a slight degradation of the quality of the centroids/clusters but in practice it is normally unnoticeable.

In contrast to the vanilla K-means, mini-batch K-means updates the centroids by taking the streaming average of previous centroids with the centroid of the mini-batch. This infuses stability in the centroids and allows the algorithm to capitalise on previous centroids calculations and throws them away after each batch update.

For the **streaming average**, we calculate the sum of the points according to the current batch (as we did in the vanilla K-means) $\sum_{\mathbf{x} \in \boldsymbol{C}_{i, \tau}} \mathbf{x}$ but then we need to multiply the previous number of points sampled so far $\bar{N}_{i}$ times the previous centroid estimate $\boldsymbol{c}_{i}$ to obtain a sum of previous points sampled so far. Then we add both sums $\bar{N}_{i} \boldsymbol{c}_{i}+\sum_{\mathbf{x} \in \boldsymbol{C}_{i, \tau}} \mathbf{x}$ and we divide them by the sum of previous and current number of points from the cluster $\bar{N}_{i}+N_{i}$. Finally, we update the number of points that have been sampled from the cluster $\bar{N}_{i}=\bar{N}_{i}+N_{i}$. This is nothing but a usual moving average.

!!! algorithm-heading "Algorithms 2: Mini-batch K-means++"

    **Input:**

    Dataset $\mathbf{X}=\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{\mathrm{N}}\right\}$ <span style="float: right;">N is the entire dataset count</span>

    $K$: The number of clusters

    $b$: Mini-batch size

    **Output**: Cluster Labels $\boldsymbol{l}=\left[l_{1}, l_{2}, \ldots, l_{N}\right] \quad l_{n} \in\{1, \ldots, K\}$

    **K-meansMB++** (**X**, *K*):

    !!! algorithm ""
        Choose $\boldsymbol{c}_{1}$ randomly

        **Repeat**

        !!! algorithm ""
            Calculate the probabilities $p\left(\mathbf{x}_{n}\right)=\frac{d\left(\mathbf{x}_{n}, \boldsymbol{c}^{\prime}\right)^{2}}{\sum_{\mathbf{x} \in \mathbf{x}} d\left(\mathbf{x}, \boldsymbol{c}^{\prime}\right)^{2}} n=1, \ldots, N$, where $\boldsymbol{c}^{\prime}$ is the nearest centroid to $\mathbf{x}_{n}$

            Sample *one* data point to be the new centroid based on probability distribution $p\left(\mathbf{x}_{n}\right)$ **until** we have chosen $K$ centroids.

            **Repeat**

            !!! algorithm ""

                **For** $\tau=1, \ldots, N / b$ <span style="float: right;"># training iterations </span>

                !!! algorithm ""

                    $\bar{N}_{i}=0 \quad i=1, \ldots, K$ <span style="float: right;"># initialise all moving sums </span>

                    **Sample** a mini-batch $\mathbf{X}_{\tau}$ of size $b$ from $\mathbf{x}$

                    **For** each $\mathbf{x}_{n}$ in $\mathbf{X}_{\tau}$ <span style="float: right;"># Assigning each data point in the batch to its closest centroid (creates local cluster) </span>

                    !!! algorithm ""

                        $l_{n}=\arg \min _{i} d\left(\mathbf{c}_{i}, \mathbf{x}_{n}\right) \quad i=1, \ldots, K$

                    **For** each local cluster $\boldsymbol{C}_{i, \tau}$ of size $N_{i}$ in batch $\mathbf{X}_{\tau}$

                    !!! algorithm ""    

                        $c_{i}=\frac{\bar{N}_{i} c_{i}+\sum_{\mathbf{x} \in C_{i, \tau}} \mathbf{x}}{\bar{N}_{i}+N_{i}}$

                        $\bar{N}_{i}=\bar{N}_{i}+N_{i}$ <span style="float: right;"># Recalculate the centroids for each global cluster by taking the streaming average to guarantee stability </span>


            **until** the centroids do not change

        **Return** the labels $\boldsymbol{l}=\left[l_{1}, l_{2}, \ldots, l_{N}\right]$

A very similar idea can be adopted to tackle clustering a stream of data, therefore we omit its details for brevity. See the apache spark example in the resources section for an idea, but bear in mind that you are not required to perform this activity, this is just for future reference.

!!! abstract "Exercise"
    See the following Jupyter Notebook exercise for a comparison of K-means and mini-batch K-means in sklearn.

    - Download exercise (.ipynb): <a href="../files/Exercise3_plot_mini_batch_kmeans.ipynb" download>Exercise 3</a>
