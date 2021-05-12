# Bisecting K-means algorithm

**Bisecting K-means algorithm takes advantage of the fact that basic 2-means clustering (as per Algorithm 1) is computationally efficient.**

Bisecting K-means algorithm makes use of the 2-means algorithm to bisect the dataset $(K-1)$ times to reach K clusters, where K>2. An additional final K-means clustering algorithm can be used to fine-tune the clusters.

The idea is as follows: In order to specify the K centroids, we can employ the idea of bisecting (splitting into 2) the dataset into two clusters. Then we pick a cluster to bisect it in turn. Each time we do the bisection in several trials (to create a **set of bisection candidates**) with random initial centroids. We then pick the pair of clusters candidates that have the **lowest SSEs** and we add them to the **list of clusters**. We keep bisecting each available cluster from the list of clusters by employing 2-means trails until we reach K-clusters. Each time we pick a cluster from the list of clusters to be bisected based on some criterion. A viable criterion is to select the cluster with the **highest SSE**. The process is repeated until we reach K-cluster.

The set of bisected K-clusters, however, may not need refinement because the splitting procedure (by using 2-means algorithm) is done by looking locally at a sub cluster and not at the level of the whole dataset. To refine further the chosen K-clusters, we choose the centroids of the bisected K-clusters as the initial centroids of a final new run of K-means algorithms, which in turn finds us the best K-clusters and their centroids. The algorithm is shown below.

!!! info "Algorithms 3: Bisecting K-means"

    **Input:**

    Dataset $\mathbf{X}=\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{\mathrm{N}}\right\}$

    $K$: The number of clusters

    Trials: number of trials to be performed in each bisection step

    **Output**: Cluster Labels $\boldsymbol{C}=\left[l_{1}, l_{2}, \ldots, l_{N}\right] \quad l_{n} \in\{1, \ldots, K\}$

    **Bisect K-means** (**X**, *K*, trials):

    !!! quote ""
        Initiate a list of clusters $C L=\left\{C_{1}\right\}$, initially cluster $C_{1}=\mathbf{X}$ <span style="float: right;"># by assigning $\boldsymbol{C}=[1,1, \ldots, 1]$ </span>

        **Repeat**

        !!! quote ""
            Remove a cluster from the list $C L$

            **For** $i=1$ to trials

            !!! quote ""

                Bisect the selected cluster using basic 2-means

            Select the two clusters from the bisection set with the lowest total SSE

            Add these two clusters to the list of clusters

         **until** the list of clusters contains $K$ clusters

         perform an extra K-means with the initial centroids of the list of clusters for fine-tuning

        **Return** the labels $\boldsymbol{C}=\left[l_{1}, l_{2}, \ldots, l_{N}\right]$

Below in figure 1.10, we show results of bisecting K-means on the problem that we mentioned previously in Figure 1.9 regarding centroid initialisation. As we can see, the algorithm is less susceptible to this problem.   

<figure role="group">
  <img src="../images/DS_IMG189.png" alt="Illustration of Bisecting K-means overcoming the issues of unlucky centroid initialisation." />
  <figcaption><strong>Figure 1.10</strong> Bisecting K-means overcoming the issues of unlucky centroid initialisation. Image reproduced from slides by Tan et al (2019), <a href="[https://www-users.cs.umn.edu/~kumar001/dmbook/index.php#item4" target="_blank">Introduction to Data Mining</a>, with kind permission of the authors.</figcaption>
</figure>    

##Density-based clustering: DBSCAN algorithm

Please refer to section 5.4.2 of Tan et al 2019.

!!! Abstract "Exercise"

    See the following tutorial in <a href="https://leeds365-my.sharepoint.com/:u:/r/personal/scsaalt_leeds_ac_uk/Documents/Downloads/Resources%20for%20ODL%20MSc/Data%20Science%20Contents/unit5/code/ClusterAnalysisTutorial.ipynb?csf=1&web=1&e=XbPlrt" target="_blank">Jupyter notebook</a> for the three types of clustering: partitional, hierarchical and spectral.
