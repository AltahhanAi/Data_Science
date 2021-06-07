#Probabilistic view of loss function for classification

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * appreciate classification from a probabilistic perspective
	  * understand the difference between generative and discriminative models.

In this section we will discuss the link between minimising a loss function and probability theory for classification, to see how learning can take a probabilistic perspective. We will also establish links with a particular probabilistic framework: the Bayesian framework for learning. This section can be safely skipped, without consequences on other sections or future sections in the module. Similar material will be also covered in some form in the Machine Learning module.

##Expected loss

For probabilistic interpretation and settings, we often want to calculate the **expected loss** instead of simple averages to take into account the fact that some data points are more **probable** than others and we would want to minimise the loss for those even on the loss of incurring a bit of loss for less probable data points. To account for this and if we are not comparing between different dataset with different sizes, we can use the simpler sum of squared error (SSE) as our loss function and the expected loss then is given by

$$
E(J)=\frac{1}{2} \sum_{n=1}^{N}\left(y\left(\mathbf{x}_{n}\right)-t_{n}\right)^{2} p\left(\mathbf{x}_{n}, t\right)
$$

This is for the discrete cases where we have N data points that are of concern. However, there is a more powerful way of coming up with a general function for the loss: that we calculate its expectation regardless of the data points that we have, and then we minimise it and then take the $\mathrm{N}$ data points that we have as samples for this continuous loss function. In simple terms, sometimes we would want to calculate the loss for a continuous infinite number of data points in the context of a model represented as a continuous function $y(\mathbf{x})$ (of course we are still going to take samples $\mathrm{N}$ ). In this case, the loss function will be written as an integral instead of the sum:

$$
E(J)=\frac{1}{2} \iint p(\mathbf{x}, t)(y(\mathbf{x})-t)^{2} \mathrm{~d} \mathbf{x} \mathrm{d} t
$$

All of the above are widely used and can be used in our treatment of training a model in general (including regression and classification).

**Note:** If we assume that all the data points have the same probability, we obtain the following expected loss called mean squared error (MAE).

$$
E(J)=\frac{1}{N} \sum_{n=1}^{N} J_{n}
$$

For the case of SSE we have

$$
E(J)=\frac{1}{N} \sum_{n=1}^{N}\left(y\left(\mathbf{x}_{n}\right)-t_{n}\right)^{2}
$$

We can also use the root mean squared error function (RMSE) which is in fact the standard deviation of the residuals. MAE and RMSE are counterparts and among the common metrics to measure the accuracy of the prediction.

$$
J=\sqrt{\frac{1}{N} \sum_{n=1}^{N}\left(y\left(\mathbf{x}_{n}\right)-t_{n}\right)^{2}}
$$

For the continuous case since the derivation cancels out one of the double integrals, we obtain:

$$
\frac{\delta E(J)}{\delta y(\mathbf{x})}=\frac{1}{2} \frac{\delta}{\delta y(\mathbf{x})}\left(\iint p(\mathbf{x}, t)(y(\mathbf{x})-t)^{2} \mathrm{~d} \mathbf{x} \mathrm{d} t\right)=\int p(\mathbf{x}, t)(y(\mathbf{x})-t) \mathrm{d} t
$$

$$
\text { Now we set } \int p(\mathbf{x}, t)(y(\mathbf{x})-t) \mathrm{d} t=0 \text { and we solve }
$$

$$
y(\mathbf{x}) \int p(\mathbf{x}, t) \mathrm{d} t-\int t p(\mathbf{x}, t) \mathrm{d} t=0
$$

$$
y(\mathbf{x})=\frac{\int t p(\mathbf{x}, t) \mathrm{d} t}{p(\mathbf{x})}=\frac{p(\mathbf{x}) \int t p(t \mid \mathbf{x}) \mathrm{d} t}{p(\mathbf{x})}
$$

$$
y(\mathbf{x})=E_{t}(t \mid \mathbf{x})
$$

This above result: $y(\mathbf{x})=E_{t}(t \mid \mathbf{x})$ is fundamental and is telling us that the best estimation $y(\mathbf{x})$ for $t$ is given as
the conditional expectation of $t$ given $\mathbf{X} .$ So from now on we need mainly to concern ourselves with this probability $p(t \mid \mathbf{x})$ as it holds the key for training our model.

##Generative, discriminative and non-probabilistic approaches

For classification problems, the above solution: $y(\mathbf{x})=E_{t}(t \mid \mathbf{x})$, suggests that we can approach the learning problem (model training) in either of the following ways (Bishop 2006):

###Approach 1:

1. Solve the inference problem of determining the joint density $p(\mathbf{x}, t)$
2. Normalise it to obtain $p(t \mid \mathbf{x})$
3. Marginalise (i.e. calculate $\left.\int t p(t \mid \mathbf{x}) \mathrm{d} t\right)$ to obtain $E_{t}(t \mid \mathbf{x})$ that is the solution for the optimisation problem

In a classification context we call models that depend on a similar approach generative models because the model that uses it can generate pairs $(\mathbf{x}, t)$ of synthetics data as per the joint density $p(\mathbf{x}, t)$.

###Approach 2:

1. Solve the inference problem of determining the joint density $p(t \mid \mathbf{x})$
2. Marginalise (i.e. calculate $\left.\int t p(t \mid \mathbf{x}) \mathrm{d} t\right)$ to obtain $E_{t}(t \mid \mathbf{x})$ that is the solution for the optimisation problem

In a classification context we call models that depend on a similar approach discriminative because the
model that uses it can discriminate whether it is likely that $t$ is the answer to a given an observation $\mathbf{x}$ as per the conditional density $p(t \mid \mathbf{x})$. But the model cannot generate synthetic data since $p(t, \mathbf{x})$ is not available.

###Approach 3: not a probabilistic

1. Find the regression function directly from the data: ex. By utilising $J=\frac{1}{2} \sum_{n=1}^{N}\left(y\left(\mathbf{x}_{n}, \boldsymbol{w}\right)-\right.$ $(y(x_n,w)-t_n )^2$ or by assuming that all the data points have the same probability.

In a classification context, models that depend on a similar approach are loosely referred to as discriminative.

##Summary

In this lesson, you have seen how to motivate a classification problem from a probabilistic perspective and how to differentiate between generative and discriminative models.

In this unit we have covered linear classification models and developed its ideas gradually to reach a non-linear multi-layer perceptron for classification.
