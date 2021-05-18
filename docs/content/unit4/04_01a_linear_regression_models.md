#Simple Linear Regression Models

**In this unit we talk about a simple yet effective group of data mining models, namely linear models.**

These are called **supervised learning techniques** since we train our models via a training set that has the required answers that we are after. Those answers are called labels. The idea is that later on after we train our model, if we present a fresh new case to it, the model will be able to guess what its answers. This brings us to two issues that we will deal with.

##What is regression?

First the nature of the answer is specified by the nature of the mining task that we are trying to perform. In supervised learning we have largely two types of answers that we are looking for: categorical or numerical. If it is categorical, our task would be to guess the category (or the class) of the cases under examination. This type of data mining task is called classification. If the label is numerical, then we need to come up with a numerical prediction that is as close as possible to the **desired answer** stored in our dataset. This type of numerical prediction is called regression. After training, during prediction, our regression model attempts to guess the best answer of an unknown data point label.


!!! note
		We call the answer that is stored in the dataset the desired answer, and the model answer is called the predicted answer. This applies for both classification and regression.

		**Remember:** we agreed to call these cases data points because each is recognised by a set of features. The features themselves can be categorical or numerical. Since we can convert any categorical data into numerical as we pointed out in unit 1, we will assume that all the data that we are dealing with is numerical.


The second question to ask is how we can verify that the model is good enough for our task; in other words, how can we measure the performance of our mode. Measuring the performance will be done by utilising the available answers that we have in our dataset. Think about it: if we do not know the correct answers, we have no way of telling whether the model answers are accurate or good enough. Therefore, we set aside part of our dataset with its available answers, for measuring the performance of our model. This is called partitioning and was covered in unit 2. So, in simple terms, similar to classification, in regression tasks, we split the dataset into two parts, one for training and one for measuring the performance. And when the data is not big enough or when we want to scrutinise the model performance, we can partition the dataset into three parts, training, validation and testing and we can further apply cross validation as we saw in unit 2 for the classification problems.

So, really the difference between classification and regression is the nature if the labels. Not that we can sometimes convert a regression problem into classification if we categorised the labels. Of course, this is not desired since we lose a lot of granularity of the labels and we only obtain very rough estimates of the answers, and usually it is not a good idea. The figure below shows a schematic illustration of the regression model, y here is a continuous value.

<figure role="group">
  <img src="../images/DS_IMG098.png" alt="Schematic illustration of regression." />
  <figcaption><strong>Figure 4.1</strong> A Schematic Illustration of regression.</figcaption>
</figure>
