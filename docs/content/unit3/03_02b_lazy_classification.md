# Lazy classification

**In decision trees we have two stages; induction and inferencing. In the inference stage, we make a prediction.**

While in the induction stage we deduce the tree itself, decision trees are called eager learners because they learn the model as soon as the data are made available to the model. The model cannot meaningfully perform the inference stage without having gone through the learning/induction process first. An alternative approach to learning is to use the lazy learning, where no learning or induction takes place. Any calculation takes place at the time of the query when prediction is required.

##Rote classification

A rote classifier is an example of lazy classification. The classifier memorises the entire dataset without learning (effectively if the dataset can be queries nothing in this stage is required). Then at the time when prediction of the label is required, rote classifier just searches for an exact match of the instances presented for classification. Then it will return the class for those instances that have matches inside the dataset. The process can be done by an SQL query if the dataset is stored in a relational database. If not, it can be simply done by filtering using any available library that can deal with data frames, such as pandas in Python. No special implementation or model building is required.

Let’s have a look at an example. Below we show a snippet of the IRIS flower dataset. The dataset was created by Ronal Fisher in the 1930s and it studies three classes of flowers: setosa, versicolor and virginica. The dataset consists of 50 samples from each of the three classes (here, we show only 3 from each 50). Four features (attributes) were measured in each sample, these are the sepal and petal width and length. Below the attributes are named with abbreviations where P stands for petal and S stands for sepal. We will call this dataset mini-iris dataset and it has a purely educational purpose. Each instance (record) is distinguished by denoting it as $\mathbf{x}_{i}, i=1, \ldots 9$.

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

Let us assume that we are presented with the following instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|2.8|4.7|1.2

In this case, the rote classifier will be able to match it directly with record 6, and the predicted label (class) will be versicolor:

Dataset|S length|S width|P length|P width|species
-----|---|---|---|---|---
$\mathbf{x}_{\mathbf{6}}=$|6.1|2.8|4.7|1.2|versicolor

Obviously, the problem with such a classifier is that if the instance does not exactly match any instance in the dataset, the classifier cannot make any decision on what would be a best match that can be utilised to label the provided instance. For example, if the classifier is presented with the following instance:

Dataset|S length|S width|P length|P width
-----|---|---|---|---
$x$= |6.1|2.8|4.7|1

It will not be able to classify the instance since there is no match for it in the dataset.
Rote classifier is an extreme case of lazy classifiers. In the next section we see something a bit less restrictive, which has much better generalisation capabilities.
