#Probabilistic view of loss function for classification

!!! success "Learning outcomes:"
	After completing this **optional** lesson you should be able to:

    * appreciate classification from a probabilistic perspective
	  * understand the difference between generative and discriminative models.

In this section we will discuss the link between minimising a loss function and probability theory for classification, to see how learning can take a probabilistic perspective. We will also establish links with a particular probabilistic framework: the Bayesian framework for learning. This lesson and its subsections provide additional information to support future modules, but it is not necessary to understand in detail in order to progress with this module. As such, you may wish to **skim read the material now, and return to it in more detail later on**.

##Expected loss

From a probabilistic perspective, we often want to calculate the **expected loss** instead of simple averages to take into account the fact that some data points are more **probable** than others and we would want to minimise the loss for them accordingly. To account for this, we start, as we did in regression, by the likelihood function.

We will discuss a case where we have a binary classification. Since we are dealing with binary classes the target $t_{n}$ can take either of two cases, 0 or 1 $t_{n} \in\{0,1\}$ in our dataset $\left\{\mathbf{x}_{n}, t_{n}\right\}$. And since we are using $y_{n}=p\left(C_{1} \mid \mathbf{x}_{n}\right)$ to calculate the probability that data point $\mathbf{x}_{n}$ is from the positive class $C_{1}$ i.e. $t_{n}=1$ (and inversely using $1-y_{n}$ to calculate the probability of $t_{n}$ being 0) we can express the probabilities of obtaining the same targets $t_{n}$ using our model parameters $\mathbf{w}$ as:

$$
p\left(t_{n} \mid \mathbf{w}\right)=\left\{\begin{aligned}
y_{n} & \text { if } t_{n}=1 \\
1-y_{n} & \text { if } t_{n}=0
\end{aligned}\right.
$$

This is a Bernoulli distribution and we can combine the expression of both cases in one function (similar to what we did for the perceptron) as follows:

$$
p\left(t_{n} \mid \mathbf{w}\right)=\left(y_{n}\right)^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

The likelihood represents the probability of all the data points being classified together and is given as:

$$
p(\mathbf{t} \mid \mathbf{w})=\prod_{n=1}^{N} p\left(t_{n} \mid \mathbf{w}\right) \mid
$$

Therefore, we have:

$$
p(\mathbf{t} \mid \mathbf{w})=\prod_{n=1}^{N}\left(y_{n}\right)^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

We would need to maximise the likelihood of our model's estimation $y_{n}$ coinciding with the actual targets $t_{n}$ of the dataset. Equivalently, we can maximise logarithm of the likelihood. And in turn we can minimise the negative of the logarithm of the likelihood which will be our loss function. In other words, our loss can be written as:

$$
J(\boldsymbol{w})=-\ln p(\boldsymbol{t} \mid \boldsymbol{w})=-\ln \prod_{n=1}^{N}\left(y_{n}\right)^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

$$
J(\boldsymbol{w})=-\sum_{n=1}^{N} t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)
$$

$$
J(\boldsymbol{w})=-\sum_{n=1}^{N} t_{n} \ln p\left(t_{n} \mid \mathbf{w}\right)+\left(1-t_{n}\right) \ln \left(1-p\left(t_{n} \mid \mathbf{w}\right)\right)
$$

As we can see this is the cross entropy that we used as our loss function in the logistic regression setting. Similarly, for the case of multi-class the loss function for the multinomial logistic regression can be motivated via the negative log of the likelihood.

##Generative, discriminative and non-probabilistic approaches

The above solution suggests that we can approach the learning problem (model training) in either of the following ways:

###Approach 1:

1. Solve the inference problem by first estimating the joint density $p(\mathbf{x}, t)$, then normalise it to obtain $p(t \mid \mathbf{x})$
3. Marginalise (i.e. calculate $\sum_{n=1}^{N} t_{n} p\left(t_{n} \mid \mathbf{x}_{n}\right)$ for regression or $\sum_{n=1}^{N} t_{n} \ln p\left(t_{n} \mid \mathbf{x}_{n}\right)$ for classification) to obtain the solution for the optimisation problem.

In a classification context we call models that depend on a similar approach **generative** models because the model that uses it can **generate** pairs $(\mathbf{x}, t)$ of synthetics data as per the joint density $p(\mathbf{x}, t)$.

###Approach 2:

1. Solve the inference problem by estimating the density $p(t \mid \mathbf{x})$ directly.
2. Marginalise (i.e. calculate $\sum_{n=1}^{N} t_{n} p\left(t_{n} \mid \mathbf{x}_{n}\right)$ for regression or $\sum_{n=1}^{N} t_{n} \ln p\left(t_{n} \mid \mathbf{x}_{n}\right)$ for classification) to obtain the solution for the optimisation problem

In a classification context we call models that depend on a similar approach **discriminative** because the
model that uses it can **discriminate** whether it is likely that $t$ is the answer to a given an observation $\mathbf{x}$ as per the conditional density $p(t \mid \mathbf{x})$. But the model cannot generate synthetic data since $p(t, \mathbf{x})$ is not available.

##Summary

In this lesson, you have seen how to motivate a classification problem from a probabilistic perspective and how to differentiate between generative and discriminative models.

In this unit we have covered linear and non-linear classification models and developed its ideas gradually to reach a non-linear multi-layer perceptron for classification.
