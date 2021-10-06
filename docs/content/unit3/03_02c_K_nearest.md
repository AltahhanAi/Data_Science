# K-nearest neighbour algorithm

**A less restrictive classifier than the rote classifier is the nearest neighbour classifier, where instead of requiring a perfect match with all the attributes of the instance, we can suffice by matching as much as possible of the attributes. In this case we just pick the instance label that best matches the provided instance.**

##Partial matching

In this case, if the classifier is presented with the previous instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|2.8|4.7|1

It will infer that the closest instance from the dataset that matches as much attributes as possible is:

Dataset|S length|S width|P length|P width|species
-----|---|---|---|---|---
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor

This dataset instance matches all but one attribute of the presented instance $x$, and the algorithm will infer that the predicted class is versicolor.

##Majority voting

But what about when we have more than one match? In this case, we can take a simple vote. For example, the following instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|2.8|5.5|1.4

Multiple matches:

Dataset |S length | S width | P length | P width | species
--------|---------|---------|----------|---------|--------
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor
$\mathbf{x}_{\mathbf{5}}=$|6.1|3|4.6|1.4|versicolor
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor
$\mathbf{x}_{\mathbf{7}}=$|6.1|2.6|5.6|1.4|virginica

In this case, it seems intuitive to take versicolor as the class since more matching instances pointed to it as the predicted class. This is called majority vote, so we look at how many instances voted with their labels and we count the labels and pick the label with the highest count. In our simple example, we have 3 versicolor against 1 virginica so versicolor wins.

##Distance metric

However, this strategy is incomplete since we might have several instances that partially match the attributes of the provided instance in different ways. For example, if the algorithm is presented with the following instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|2.5|4.5|1.4

There are two records that match the first and last attributes, these are:

Dataset |S length | S width | P length | P width | species
--------|---------|---------|----------|---------|--------
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor
$\mathbf{x}_{\mathbf{7}}=$|6.1|2.6|5.6|1.4|virginica

The question is then which matching instance should we choose to be the label of the provided instance versicolor or virginica? This poses the question of which one of these matching instances is closest to the presented instance. If we can specify this quantitatively then we pick the instance from the dataset that are the nearest to the provided instance. This means that we need to utilise a distance that provides quantitative information about how close each instance in the dataset from the provided instance. We can then pick the lowest distance to match best with the provided instance and we pick its label to be the label of the provided instance. For the above example, we can adopt the usual Euclidian distance. And to avoid unnecessary calculations of the square roots we can compare the squared Euclidian distances (since taking the square root will not change the result of the comparison). In this case, the squared distances are:

$$
\begin{array}{l}
d^{2}\left(\mathbf{x}, \mathbf{x}_{4}\right)=(6.1-6.1)^{2}+(2.5-2.9)^{2}+(4.5-4.7)^{2}+(1.4-1.4)^{2}=0.2 \\
d^{2}\left(\mathbf{x}, \mathbf{x}_{7}\right)=(6.1-6.1)^{2}+(2.5-2.6)^{2}+(4.5-5.6)^{2}+(1.4-1.4)^{2}=1.2
\end{array}
$$

This means that x_4 is closer to the presented instance than $\mathbf{X}_{7}$ and the predicted label (class) will be the class of $\mathbf{X}_{4}$ i.e. versicolor.

##Metric distance for nearest neighbours

We can employ any of the metrics or similarity measures that we have talked about in the previous lesson. Each metric gives a different algorithm power that suits specific problems. For example, we can use cosine similarity to classify documents using nearest neighbours.

But here again, conceivably, we might have several instances that have the same distance from the provided instance and they have different labels. What would be the label of the provided instance in this case?

For example, if the algorithm is presented with the following instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|3.5|4.7|0.2

Think about it for a moment… Ok, yes, sure we can just take a count/vote of how many instances have label 1 and how many have label 2, and we take the label that has the highest count.

Dataset |S length | S width | P length | P width | species
--------|---------|---------|----------|---------|--------
$\mathbf{x}_{\mathbf{1}}=$|5.1|3.5|1.4|0.2|setosa
$\mathbf{x}_{\mathbf{3}}=$|5.5|3.5|1.3|0.2|setosa
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor

##Distance-weighted voting

However, since we introduced the distances, this is an incomplete approach. Although there might not be a direct match between some of the features of the presented instance and the features of the instances in the dataset, but they might still be close to each other. In reality, we would need to take the distances for the entire dataset and then we compare. This by itself can lead to some issues such as efficiency of the technique but it is necessary for numerical dataset. But the more serious issue is that we cannot take the vote of the entire dataset as it is always going to give us the same results based on the frequency of the labels themselves.

A remedy for this issue is to take the vote of the nearest $k$ instance to the presented instance. $k$ is a hyper parameter that has an important effect on the predictions of the algorithm. In our case if we take $k=4$, we get the following nearest neighbours for the presented instance:

Dataset |S length | S width | P length | P width | species |$d^{2}\left(\mathbf{x}, \mathbf{x}_{i}\right)$
--------|---------|---------|----------|---------|--------|---
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor|1.49
$\mathbf{x}_{\mathbf{5}}=$|6.1|3|4.6|1.4|versicolor|1.7
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor|1.8
$\mathbf{x}_{\mathbf{8}}=$|6.1|3|4.9|1.8|virginica|2.85

And hence we see that the setosa class is not a participant, since all of its instances are much further than the shown instance above. Therefore, taking the vote of this nearest neighbour algorithm gives us versicolor as the class.

But again what if the voting instance casted exactly the same count for both labels? For example, choosing $k=6$ gives us the following equal voting between versicolor and virginica:

Dataset |S length | S width | P length | P width | species |$d^{2}\left(\mathbf{x}, \mathbf{x}_{i}\right)$
--------|---------|---------|----------|---------|--------|---
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor|1.49
$\mathbf{x}_{\mathbf{5}}=$|6.1|3|4.6|1.4|versicolor|1.7
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor|1.8
$\mathbf{x}_{\mathbf{8}}=$|6.1|3|4.9|1.8|virginica|2.85
$\mathbf{x}_{\mathbf{7}}=$|6.1|2.6|5.6|1.4|virginica|3.06
$\mathbf{x}_{\mathbf{9}}=$|6.5|3|5.5|1.8|virginica|3.61

In this instance we can weigh the votes according to the inverse of their distances, the closer the instance from the dataset to the provided instance, the higher its weight. We multiply the weights with the votes, and we sum the weights to give us a quantification of which label should win to be the best label for the provided instance.

Dataset |S length | S width | P length | P width | species |$d^{2}\left(\mathbf{x}, \mathbf{x}_{i}\right)$ | weight
--------|---------|---------|----------|---------|--------|---|---
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor|1.49|0.67
$\mathbf{x}_{\mathbf{5}}=$|6.1|3|4.6|1.4|versicolor|1.7|0.59
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor|1.8|0.56
$\mathbf{x}_{\mathbf{8}}=$|6.1|3|4.9|1.8|virginica|2.85|0.35
$\mathbf{x}_{\mathbf{7}}=$|6.1|2.6|5.6|1.4|virginica|3.06|0.33
$\mathbf{x}_{\mathbf{9}}=$|6.5|3|5.5|1.8|virginica|3.61|0.28

**weight sum** for versicolor: 1.82

**weight sum** for virginica: 0.96

In this case the algorithm predicts that the class of the presented instance is versicolor.

OK, what if the sum of the weights turns out to be the same as you might have pre-empted? Well in this case, we just can pick any label since both have same matching powers for the provided instance. But wait, we can do something else, we can change the distance metric that is used to provide a different perspective on the weights and then pick the label that scored best in two or more measures. However, we will not cover this last point and we suffice by assuming that the designer will pick the best distance measure to match the nature and quality of the dataset under consideration.

##Rescaling to improve performance

There is one more piece here to complete our example coverage of kNN. You might have noticed that it was a bit odd that the virginica class should dominate, since its attribute seems to be closer than versicolor. This is in fact a result of one attribute or feature dominating other features. In particular, note that the PWidth feature is small in comparison with SLength. Hence the contribution of SLength is exaggerated in the calculations and its effect is magnified unnecessarily. We can overcome this phenomena, however, by a simple trick: rescaling. By rescaling, every feature would have its value ranged between [0, 1] which guarantees better overall performance for the kNN algorithm. In fact, being susceptible to the dominance of some features is one of the limitations of kNN. Here, we should overcome this issue by pre-processing.

Below, we show the rescaling of the mini-iris where we divided each attribute by its maximum. Note how rescaling cuts across instances vertically on the dataset.

###Mini-iris

Dataset |S length | S width | P length | P width | species
--------|---------|---------|----------|---------|--------
$\mathbf{x}_{\mathbf{1}}=$|5.1|3.5|1.4|0.2|setosa
$\mathbf{x}_{\mathbf{2}}=$|4.9|3|1.4|0.2|setosa
$\mathbf{x}_{\mathbf{3}}=$|5.5|3.5|1.3|0.2|setosa
$\mathbf{x}_{\mathbf{4}}=$|6.1|2.9|4.7|1.4|versicolor
$\mathbf{x}_{\mathbf{5}}=$|6.1|3|4.6|1.4|versicolor
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor
$\mathbf{x}_{\mathbf{7}}=$|6.1|2.6|5.6|1.4|virginica
$\mathbf{x}_{\mathbf{8}}=$|6.1|3|4.9|1.8|virginica
$\mathbf{x}_{\mathbf{9}}=$|6.5|3|5.5|1.8|virginica


###Normalised mini-iris (only two decimal points are shown)

Dataset |S length | S width | P length | P width | species
--------|---------|---------|----------|---------|--------
$\mathbf{x}_{\mathbf{1}}=$|0.78|1.00|0.25|0.11|setosa
$\mathbf{x}_{\mathbf{2}}=$|0.75|0.86|0.25|0.11|setosa
$\mathbf{x}_{\mathbf{3}}=$|0.85|1.00|0.23|0.11|setosa
$\mathbf{x}_{\mathbf{4}}=$|0.94|0.83|0.84|0.78|versicolor
$\mathbf{x}_{\mathbf{5}}=$|0.94|0.86|0.82|0.78|versicolor
$\mathbf{x}_{\mathbf{6}}=$|0.94|0.80|0.84|0.67|versicolor
$\mathbf{x}_{\mathbf{7}}=$|0.94|0.74|1.00|0.78|virginica
$\mathbf{x}_{\mathbf{8}}=$|0.94|0.86|0.88|1.00|virginica
$\mathbf{x}_{\mathbf{9}}=$|1.00|0.86|0.98|1.00|virginica

Now, if we take the k=6 nearest neighbour, we get a different result which is as follows:

Dataset |S length | S width | P length | P width | species |$d^{2}\left(\mathbf{x}, \mathbf{x}_{i}\right)$ | weight
--------|---------|---------|----------|---------|--------|---|---
$\mathbf{x}_{\mathbf{6}}=$|0.94|0.80|0.84|0.67|versicolor|0.349|2.87
$\mathbf{x}_{\mathbf{1}}=$|0.78|1.00|0.25|0.11|setosa|0.371|2.7
$\mathbf{x}_{\mathbf{3}}=$|0.85|1.00|0.23|0.11|setosa|0.377|2.65
$\mathbf{x}_{\mathbf{2}}=$|0.75|0.86|0.25|0.11|setosa|0.402|2.49
$\mathbf{x}_{\mathbf{5}}=$|0.94|0.86|0.82|0.78|versicolor|0.465|2.15
$\mathbf{x}_{\mathbf{4}}=$|0.94|0.83|0.84|0.78|versicolor|0.475|2.11

**weight sum k=6** for versicolor: 2.87  +  2.15  +  2.11 =  7.13

**weight sum k=6** for setosa: 2.7  +  2.65  +  2.49  =  7.84

This means that eventually the algorithm should classify the given instance as setosa.

##kNN algorithm

To summarise, in the k-nearest neighbour classification technique, we look at $k$ of the nearest neighbours to the presented data point and we take a simple (possibly weighted) vote to classify the data point $x$ based on the classes of its closest neighbours $\mathbf{y}_{i} i=1, \ldots k$. We must consider rescaling as kNN is susceptible to issues related to feature dominance. Note that this is a classification techniques, i.e. there are a set of $K$ labels representing the classes that are already provided in the dataset (differentiate between $k$ and $K$). Unlike clustering, where we do not have these labels available and instead the algorithms will come up with cluster labels to represent the clusters. kNN can produce decision boundaries of arbitrary shape. This gives it a lot of power to express complex problems, albeit a simple algorithm by nature. In comparison with decision trees, kNN gives far more flexibility of representing complex decision boundaries. kNN face difficulties when dealing with missing values for some of the attributes. Irrelevant features can distort some important distance metric which in turn reduce the effectiveness of kNN. The kNN algorithm is shown below.

!!! algorithm-heading "Algorithms 4: k-nearest neighbours"

    **Input:**

    Input training set $\mathbf{X}=\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{\mathrm{N}}\right\}$

    Corresponding target labels $\boldsymbol{t}=\left\{t_{1}, t_{2}, \ldots, t_{N}\right\}$

    Input test set $\mathbf{X}^{\prime}=\left\{\mathbf{x}_{1}^{\prime}, \mathbf{x}_{2}^{\prime}, \ldots, \mathbf{x}_{S}^{\prime}\right\}$ S is the size of the test set.

    $k$: The number of neighbours.

    **Output**: Class Labels $\boldsymbol{C}=\left\{y_{1}, y_{2}, \ldots, y_{S}\right\}$ for the test set $\mathbf{X}^{\prime}$ where $y_{n} \in\left\{l_{1}, \ldots, l_{K}\right\}$

    $\mathbf{k N N}\left(\mathbf{X}, \boldsymbol{t}, \mathbf{X}^{\prime}, k\right)$:

    !!! algorithm ""

        **For** each $\mathbf{x}_{i}^{\prime}$ in $\mathbf{X}^{\prime}$

        !!! algorithm ""

            **For** each $\mathbf{x}_{j}$ in $\mathbf{X}$

            !!! algorithm ""

                **Compute** $d\left(\mathbf{x}_{i}^{\prime}, \mathbf{x}_{j}\right)=\mathbf{x}_{j}^{\top} \mathbf{x}_{i}^{\prime}$<span style="float: right;"># or other appropriate distance for the problem in hand</span>

                **Select** the set of $k$ nearest neighbours $\boldsymbol{D}_{i}$ for $\mathbf{X}_{i}^{\prime}$

                $y_{i}=\max _{l} \sum_{\left(\mathrm{x}_{j}, t_{j}\right) \in \boldsymbol{D}_{i}} I\left(t_{j}=l\right)$            

        **return** $\left\{y_{i}\right\} \quad i=1, \ldots, S$
