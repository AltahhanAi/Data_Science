#Multi-class problems

In this lesson we will extend the binary classifiers that we have covered previously, the perceptron and the logistic regression, into multi-class problems. We can do that in several ways. By combining multiple of these binary classifiers or by adjusting the basic structure of the techniques, it can deal with multiple classes. Both have similarities and advantages and disadvantages that we will discuss.

## 1-OF-K BINARY CODING FOR NUMERICAL TECHNIQUES

In the former, since we are dealing with a progression of performance categories, then we can encode the labels as $\{3,2,1\} .$ In this case the label will take one and only one of the $\{3,2,1\}$ labels and the output of the prediction have one component $t_{n}=(C) .$ The estimated classes might take something in between and either we interpret the predicted class values that lies in-between as a degree of closeness to the class. So for example if we get 2.2 we interpret it as a value between ‘Medium’ and ‘High’ and being closer to Medium. Or we apply a threshold to round the result to its nearest integer. So for example if we get 2.2 we interpret it as 2 i.e. medium and so on.

On the other hand, for the latter set of labels {‘Truck’, ‘Sedan’, ‘SUV’}, it makes more sense to just utilise a 0,1 scheme to represent ‘Truck’ and ‘no Truck’ class and the same for other classes, where we have also 0,1 represents  ‘Sedan’ ‘no Sedan’ and 0,1 represents  ‘SUV’ ‘no SUV’. To do so, we would need to make our classifier out put three values $t_{n}=\left[C_{1}, C_{2}, C_{3}\right]$ each $C_{i}$ can take either 0 or $1 .$ Such a scheme would get us into issues of having no class at all when all of them are 0 s or predicting more than one class for the same input,so to avoid these issues we implement a 1 -of-K binary coding scheme where we allow one and only one of the $\mathrm{Ci}$ values to take the value of 1 and the rest must all be 0 s. Note that we have been dealing with this scheme for a binary class problem when we assumed that a one class takes the value of 0 and the other takes the value of 1 .
This is the most common scheme for classification, however other codings are possible. For example the perceptron (which deals with binary class problem but can be extended to multi-classes) uses a $\{1,-1\}$ coding for
the two classes.

##Issues of combing a set of independent binary classifiers to deal with a multi-class problem

When dealing with multi-class problems using a set of binary classifiers we might be tempted to use one for each class independently and then combine them in some way. However this can lead to some serious issues as we show in figure 5.1 below.

<figure role="group">
  <img src="../images/DS_IMG156.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.1:</strong> similar to Bishop 2006. (Left): two binary classifiers used independently in a one-versus-all (OvA) fashion to decide whether an instance belongs or does not belong to a class. (Right): Binary classifiers used independently in one-versus-one fashion to decide if an instance belongs to one of two classes.</p>
  </figcaption>
</figure>

In figure 5.1 above, the yellow area to the left is an area of ambiguity where the two classifiers can dictate that an
instance belongs to both class $C_{1}$ and class $C_{2}$. The yellow area to the right is an area of ambiguity where an instance can be decided to belong to two or three classes at the same time. The solution is to use a classifier with
multi-linear boundaries $y_{k}(x)$ (similar to when we had multi-output regression) and decide on an instance x belongs to class $C_{k}$ only when its activation function $y_{k}(x)$ is higher than all other classes activations $y_{j}(x) .$ One way to do that is by normalising the regularised scores that we obtain from each independent classifier (on the left) and instead of applying one-versus-all approach we choose the highest score. You will see some examples of the effectiveness of this strategy later.

##Multi-output (multi-class) linear classification models

The structure would be similar to the structure that we have seen for the multi-output linear regression which is shown below for convenience.

<figure role="group">
  <img src="../images/DS_IMG157.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.2: Schematic representation of a multi-output independent logistic regression linear classifiers with basis.</strong></p>
  </figcaption>
</figure>

The linear classification model with multiple output can be expressed as

$$
\boldsymbol{y}(\mathbf{x}, \mathbf{W})=g\left(\mathbf{W}^{\top} \boldsymbol{\phi}(\mathbf{x})\right)
$$

Figure 5.2 above shows logistic regression with $K$ multiple outputs. The number of outputs can be associated with multiple classes $K$ where in general $\hat{K} \geq K-1$ with $\hat{K}=K-1$ when the classes are linearly separable and their boundaries are parallel. For example we may need only 2 lines to separate 3 classes. When the classes are non-linearly separable, $K$ corresponds to the number of decision boundaries needed to separate the classes. The issue with this structure is that it does not harmonise the outputs and may run into the issues mentioned in the previous section (particularly the issues associated with one-versus-all) because of the independence of the outputs. In such cases, further processing will be needed in order to decide which class the data point is from.
Each classifier $i$ is specialised in one class $i$ and produces an independent probability estimate $p_{i}\left(C_{i} \mid \mathbf{x}\right) .$ One of the simplest approaches is to normalise the scores $p_{i}\left(C_{i} \mid \mathbf{x}\right)$ that were produced by the independent classifier to properly obtain a unified probability distribution $p\left(C_{i} \mid \mathbf{x}\right)=\frac{p_{i}\left(C_{i} \mid \mathbf{x}\right)}{\sum_{i=1}^{K} p_{i}\left(C_{i} \mid \mathbf{x}\right)}$ over the different classes $i$. The final decision on which class the data point is from can be performed by picking the class with the max probability; see section 5.2 of <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7457&rep=rep1&type=pdf" target="_blank">this paper by Zadrozny and Elkan (2002)</a> for more details. See the following <mark>Jupyter notebook</mark> for more on classifying iris dataset 3 classes, and on how a one-versus-all approach creates an ambiguous triangular decision region. See also <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html" target="_blank">here</a> and <a href="https://scikit-learn.org/stable/modules/sgd.html#sgd" target="_blank">here</a> to know more about linear classifiers in sklearn.

The same structure can be produced to the perceptron we only need to change the activation function.

<figure role="group">
  <img src="../images/DS_IMG158.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.3: Schematic representation of a multi-output independent perceptron linear classifiers with basis.</strong></p>
  </figcaption>
</figure>

<mark>Have tried to interpret figure number correctly below. Typos in 5.4-5.7</mark>

<figure role="group">
  <img src="../images/DS_IMG159.png" alt="Brief description." />
  <img src="../images/DS_IMG160.png" alt="Brief description." />
  <img src="../images/DS_IMG161.png" alt="Brief description." />
  <img src="../images/DS_IMG162.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figures 5.4-5.7: Decision boundaries for multi-output (multi-class) perceptron (left) and logistic regression (right) for the iris dataset.</strong> The setosa is linearly separable from the rest, while versicolor and virginica are non-linearly separable. (Top) shows one-versus-all (OvA) which creates ambiguous regions (hyperplanes) shown by the dashed lines. For the perceptron it is struggling to come up with close enough boundaries and thrown off by the outliers in both the versicolor and the virginica. The logistic regression one-versus-all did better in that sense because it is much more resilient towards outliers, but it still has an ambiguous region (the triangle in the middle of the figure) where two classifiers are thinking that the data in the middle belongs to their respective positive class. (Bottom) The shaded decision regions are obtained via the normalisation of the scores given by the independent classifiers in order to overcome the ambiguity issue of the OvA approach that we mentioned in the previous section.</p>
  </figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG163.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.8: Decision boundaries for multi-output (multi-class) both for logistic regression on the iris dataset.</strong> Left shows a multinomial logistic regression while the right shows normalised multi-output logistic regression. The difference is that in the left we normalise an exponential activation functions for multi-output, on the right we normalise a multiple logistic functions instead of exponential functions. As you can see, multinomial logistic regression deals better with ambiguity but it is still there because essentially we are still dealing with linear models. We need a more complex model to deal with issue such as a neural network that is capable of generating a curved shape boundaries.</p>
  </figcaption>
</figure>

Watch a video2 that explains the above concepts.

DS-VID-17

<mark>Video needed</mark>

##Multinomial logistic regression: Softmax regression

In the last section we saw how logistic regression has been built originally for binary classification where we expect the output of the activation function to give us one value in the range $[0,1]$. The value represents the probability of the input being from the positive class and we get the probability from the negative class by exploiting that the sum of both must be 1 . In other words, if $p\left(C_{1} \mid \mathbf{x}\right)$ is given by the one output of the logistic regression model then $p\left(C_{0} \mid \mathbf{x}\right)=1-p\left(C_{1} \mid \mathbf{x}\right)$

We also saw how we can adapt multiple of these binary classifiers in order to deal with multi-class problem where we have multiple classes that we need to deal with. Basically, we exploited the multi-output structure that we developed in regression and we extended it to multi binary classification to deal with multi-class cases. We have seen three approaches to harmonise or combine the multi-output into one decision: where we can apply OvA or OvO or combine in other ways. We saw also that OvA and OvO approaches create ambiguous decision regions and we saw that that we can deal OvA ambiguity via normalisation.

In this section we will extend the structure of logistic regression into multi-classes. The technique is called multinomial logistic regression because we are dealing with multinomial target variable. Binomial and multinomial are names that comes from probability theory to deal with variables that takes only two values or multiple values. The dependent variable here is the label which can take one of a multiple categorical values. So essentially it is a fancy name for multi-class problems. So we are extending logistic regression from binomial to multinomial. We will start with binary logistic regression that we covered in a previous section.

Let us assume that we build a multi-output model as shown in figure 5.9:

<figure role="group">
  <img src="../images/DS_IMG164.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.9: 2-output linear classifier with exponential activation function.</strong></p>
  </figcaption>
</figure>

$$
\begin{array}{l}
y_{1}=e^{\mathbf{w}_{1}^{\top} \boldsymbol{\phi}} \\
y_{2}=e^{\mathbf{w}_{2}^{\top} \boldsymbol{\phi}}
\end{array}
$$

Let us normalise

$$
\bar{y}_{1}=\frac{e^{\mathrm{w}_{1}^{\top} \phi}}{e^{\mathbf{w}_{1}^{\top} \phi}+e^{\mathbf{w}_{2}^{\top} \phi}}=\frac{1}{1+\frac{e^{\mathbf{w}_{2}^{\top} \phi}}{e^{\mathbf{W}_{1}^{\top} \phi}}}=\frac{1}{1+e^{\mathbf{w}_{2}^{\top} \phi-\mathbf{w}_{1}^{\top} \phi}}=\frac{1}{1+e^{\left(\mathbf{w}_{2}^{\top}-\mathbf{w}_{1}^{\top}\right) \phi}}=\frac{1}{1+e^{\left(\mathbf{w}_{2}-\mathbf{w}_{1}\right)^{\top} \phi}}
$$

By defining $\mathbf{w}:=\mathbf{w}_{2}-\mathbf{w}_{1}$ and substituting we get:

$$
\bar{y}_{1}=\frac{1}{1+e^{\mathbf{w}^{\top} \phi}}
$$

Similarly due to normalisation we have:

$$
{\bar{y}}_2=1-{\bar{y}}_1
$$

So we have proved that: ${\bar{y}}_1=y$ and ${\bar{y}}_2=1-y$

In other words, we have proven that we can move from the 2-output model to the logistic regression model by normalisation.

We call the normalised exponential activation function a **softmax** function (or a Gibbs or Boltzmann function when we use a cooling parameter with it –originally used in thermodynamic literature- you will come across this function later in the Deep Learning module).

Softmax is a generalisation of the logistic function into multi-dimensional space. Note that we get the same result if we define $y_1=e^{-\mathbf{w}_1^\top{\phi}}$ and  $y_2=e^{-\mathbf{w}_2^\top{\phi}}$ we just get that ${\bar{y}}_2=\frac{1}{1+e^{\mathbf{w}^\top{\phi}}}$ and that ${\bar{y}}_1=1-{\bar{y}}_2$ which is essentially the same result.

With softmax we need to calculate first the exponential activation for all the outputs and then we normalise them, so there is the added overhead of normalisation. This is not a problem here, given that we are talking about small number of classes. But when the structure needed to be parallelised, the normalisation creates a bottleneck if it is in the middle of a larger structure or if it is in a hidden layer (although it is rare to use softmax in the hidden layers). So, it is faster to use logistic function directly for the 2-class problems instead of the softmax.

In figure 5.10 below we show the two output linear classification function with a softmax activation function and its equivalent logistic regression structure. We signify the softmax by adding a dashed box around the outputs and a bar on top of each output like ${\bar{y}}_1$ and ${\bar{y}}_2$. This represents the normalisation operation. Note the difference between scaling and normalisation. **Scaling** a component of all instances (dashed line box) is a pre-processing operation and is performed on the level of the entire dataset. **Normalising** an instance (fine dotted line box) is performed on the level of individual instance. In simple terms, in scaling we take a max of each a component in a tabular dataset and we divide all instances of the component by this max. In normalisation we go across the components of an individual instance and we divide by the sum of the components of this instance.

<figure role="group">
  <img src="../images/DS_IMG165.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.10: 2-output normalised linear classifier with exponential activation function (left) and its equivalent logistic regression (right).</strong></p>
  </figcaption>
</figure>

So, now we are in a position to be able to generalise the logistic regression of a binary class problem into a multi-class problems by just defining a structure with as many outputs as we want with an exponential activation function that we normalise.

<figure role="group">
  <img src="../images/DS_IMG166.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 5.11:</strong> Multinomial Logistic Regression (aka softmax regression): a multi-output normalised linear classifier with exponential activation function. Note that if we have K classes \acute{K} can be made = K-1 instead of K if we exploit that one of the normalised output is unnecessary as it can be deduce from summation of the outputs to 1 (called pivoting).</p>
  </figcaption>
</figure>

$$
\bar{y}_{k}=\frac{y_{k}}{\sum_{i=1}^{K} y_{i}}=\frac{e^{\mathbf{w}_{k}^{\top} \phi}}{\sum_{i=1}^{K} e^{w_{l}^{\top} \phi}}
$$

Finally the multinomial logistic regression model can be succinctly defined as

$$
\boldsymbol{y}(\mathbf{x}, \mathbf{W})=e\left(\mathbf{W}^{\top} \boldsymbol{\phi}(\mathbf{x})\right)
$$

$$
\bar{y}=\frac{1}{\sum_{l=1}^{K} y_{i}} y
$$

## Loss Function
Recall that we have used the cross entropy for logistic regression where we have

$$
\tilde{H}_{n}(\mathbf{w})=\underbrace{-t_{n} \log \left(y_{n}\right)}_{\text {related to class } c_{1}}\underbrace{-\left(1-t_{n}\right) \log \left(1-y_{n}\right)}_{\text {related to class } c_{0}}
$$

This form implicitly chooses the one of the two terms $\log \left(y_{n}\right)$ or  $\log \left(1-y_{n}\right)$ depending on the actual class. If the class is $t_n=0$ then the first term cancels and we are left with the second term $\log \left(1-y_{n}\right)$, while if the class is $t_n=1$ the second term cancels and we are left with the first term $\log \left(y_{n}\right)$.
We can apply a more general form of cross entropy to multi-class by adopting a 1-of-k binary coding for the label. The resultant mean over the whole dataset will be used as the loss function for a softmax regression for a problem with multi-class.


Remember that if we use 1-of-k binary coding for the label, then each label takes the form ${t}_n=[0,\ 0,\ldots,1,0,\ldots,0]$. So for example if the data point ${x}_n$ actual class $C_3$ and we have a total of 4 classes in the problem, then ${t}_n=\left[0,\ \ 0,\ 1,\ 0\right]$ while if point $\mathbf{x}_n$ actual class is $C_1$ then ${t}_n=\left[1,\ 0,\ 0,\ 0\right]$ and so on. So if we assume that we uses 1-of-k binary coding then the cross entropy of a multi-class problem can be defined as:

$$
\tilde{H}_{n}\left(\mathbf{w}_{k}\right)=\left\{\begin{array}{cl}
-\log \left(\bar{y}_{k}\left(x_{n}\right)\right) & \text { when } t_{k, n}=1 \\
0 & \text { when } t_{k, n}=0
\end{array}\right.
$$

Which can be written succinctly as:
<mark>Equations below not coming out right still don't fully match script</mark>

$$
\tilde{H}_{n}\left(\mathbf{w}_{k}\right)=-\sum_{k=1}^{K} t_{k, n} \log \left(\bar{y}_{k}\left(x_{n}\right)\right)
$$

The derivatives also satisfy the nice property that we described in the logistic regression section. You can choose to see how to derive the gradient in the box at the end of this section, but you can carry on with the rest of the unit without doing so, this is not assessed.

$$
\nabla_{k} \widetilde{H}_{n}\left(\mathbf{w}_{k}\right)=-\nabla_{k} \sum_{k=1}^{K} t_{k, n} \log \left(\bar{y}_{k}\left(\boldsymbol{x}_{n}\right)\right)
$$

$$
\nabla_{k} \widetilde{H}_{n}\left(\mathbf{w}_{k}\right)=-\boldsymbol{\phi}_{n}\left(t_{k, n}-\bar{y}_{k, n}\right)
$$

where we denoted $\nabla_k≔∇wk$. The gradient is again taking the same form of a linear regression data point error. This is great as the algorithms will take a very similar shape as we saw earlier for the logistic regression.

And so the stochastic gradient decent update takes the usual form of the logistic regression but for each normalised output ${\bar{y}}_{k,n}={\bar{y}}_k(\mathbf{x}_n)$ as follows:

$$
\mathbf{w}_{k}^{(\tau+1)}=\mathbf{w}_{k}^{(\tau)}+\eta \phi_{n}\left(t_{k, n}-\bar{y}_{k, n}\right)
$$

$$
\mathbf{w}_{k}^{(\tau+1)}=\mathbf{w}_{k}^{(\tau)}+\eta \boldsymbol{\phi}_{n}\left(t_{k, n}-f\left(\mathbf{w}^{\top} \boldsymbol{\phi}_{n}\right)\right)
$$

And the loss function for the softmax regression can be written as:

$$
\bar{J}=\frac{1}{N} \sum_{n=1}^{N} \widetilde{H}_{n}\left(\mathbf{w}_{k}\right)
$$

$$
\bar{J}=-\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} t_{k, n} \log \left(\bar{y}_{k}\left(\boldsymbol{x}_{n}\right)\right)
$$

The gradient takes the form:

$$
\nabla_{k} \bar{J}=-\frac{1}{N} \sum_{n=1}^{N} \nabla_{k} \widetilde{H}_{n}\left(\mathbf{w}_{k}\right)
$$

$$
\nabla_{k} \bar{J}=-\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{\phi}_{n}\left(t_{k, n}-\bar{y}_{k, n}\right)
$$

### The Gradient of the Cross Entropy Terms from Multi Class Problem
	We define  \sum_{i=1}^{K}y_i≔y  and hence {\bar{y}}_k=\frac{y_k}{\sum_{i=1}^{K}y_i}=\frac{y_k}{\bar{\bar{y}}}
	Also we note that \sum_{k=1}^{K}t_k=1
	We will denote \nabla_{\mathbf{w}_j}≔∇j 	  where we have	  y_j=e^{\mathbf{w}_j^\top\mathbit{\phi}}\ \
	\nabla_{\mathbf{w}_j}y_j=\ \nabla_jy_j=\mathbit{\phi}e^{\mathbf{w}_j^\top\mathbit{\phi}}=\mathbit{\phi}y_j
	\nabla_jy_k=0 when j\neq k
	\nabla_j\sum_{i=1}^{K}y_i=\mathbit{\phi}y_j\ \ and\ \ \ \nabla_j\bar{\bar{y}}=\mathbit{\phi}y_j
	\nabla_j\log{y_j}=\frac{\nabla_jy_j}{y_j}=\frac{\mathbit{\phi}y_j}{y_j}=\mathbit{\phi}
Now we are ready to get the derivative
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\nabla_j\sum_{k=1}^{K}{t_k\log{\left({\bar{y}}_k\right)}} 				
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\nabla_j\sum_{k=1}^{K}{t_k\log{\left(\frac{y_k}{\bar{\bar{y}}}\right)}}  			as per 1
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\nabla_j\left(\sum_{k=1}^{K}{t_k\log{y_k}}-\sum_{k=1}^{K}{t_k\log{\bar{\bar{y}}}}\right) 		
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\nabla_j\left(\sum_{k=1}^{K}{t_k\log{y_k}}-\log{\bar{\bar{y}}}\sum_{k=1}^{K}t_k\right)		 
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\nabla_j\left(\sum_{k=1}^{K}{t_k\log{y_k}}-\log{\bar{\bar{y}}}\right) 		as per 2
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\left(\sum_{k=1}^{K}{t_k\nabla_j\log{y_k}}-\nabla_j\log{\bar{\bar{y}}}\right)			
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\left(t_j\mathbit{\phi}_n-\frac{\mathbit{\phi}y_j}{y}\right)			as per 3.c and 3.d
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\left(t_j\mathbit{\phi}_n-{\bar{y}}_j\right)
\nabla_j{\widetilde{H}}_n\left(\mathbf{w}_j\right)=-\mathbit{\phi}_n\left(t_j-{\bar{y}}_j\right)

## 5.6	REGULARISED MINI-BATCH STOCHASTIC GRADIENT DESCENT UPDATES FOR MULTI-CLASS LOGISTIC REGRESSION MODEL

A regularised mini-batch stochastic gradient decent can be devised for this technique as we did earlier and is shown below.

Note that for the logistic regression there is no least squares solution, as this does not suit the loss function which was based on the entropy. It can be proven that the cross entropy is equivalent to a maximum likelihood approach.

!!! algorithm "xxx"
