#Multi-class problems

In this lesson we will extend the binary classifiers that we have covered previously, the perceptron and the logistic regression, into multi-class problems. We can do that in several ways. By combining multiple of these binary classifiers or by adjusting the basic structure of the techniques, it can deal with multiple classes. Both have similarities and advantages and disadvantages that we will discuss.

In the former, since we are dealing with a progression of performance categories, then we can encode the labels as $\{3,2,1\} .$ In this case the label will take one and only one of the $\{3,2,1\}$ labels and the output of the prediction have one component $t_{n}=(C) .$ The estimated classes might take something in between and $\underline{\underline{\text { either we interpret }}}$ the predicted class values that lies in-between as a degree of closeness to the class. $\underline{\text { So }}$ for example if we get $2.2$ we interpret it as a value between ‘Medium’ and ‘High’ and being closer to Medium. Or we apply a threshold to round the result to its nearest integer. So for example if we get 2.2 we interpret it as 2 i.e. medium and so on.

On the other hand, for the latter set of labels {‘Truck’, ‘Sedan’, ‘SUV’}, it makes more sense to just utilise a 0,1 scheme to represent ‘Truck’ and ‘no Truck’ class and the same for other classes, where we have also 0,1 represents  ‘Sedan’ ‘no Sedan’ and 0,1 represents  ‘SUV’ ‘no SUV’. To do so, we would need to make our classifier out put three values $t_{n}=\left[C_{1}, C_{2}, C_{3}\right]$ each $C_{i}$ can take either 0 or $1 .$ Such a scheme would get us into issues of having no class at all when all of them are 0 s or predicting more than one class for the same input,so to avoid these issues we implement a 1 -of-K binary coding scheme where we allow one and only one of the $\mathrm{Ci}$ values to take the value of 1 and the rest must all be 0 s. Note that we have been dealing with this scheme for a binary class problem when we assumed that a one class takes the value of 0 and the other takes the value of 1 .
This is the most common scheme for classification, however other codings are possible. For example the perceptron (which deals with binary class problem but can be extended to multi-classes) uses a $\{1,-1\}$ coding for
the two classes.

##Issues of combing a set of independent binary classifiers to deal with a multi-class problem

When dealing with multi-class problems using a set of binary classifiers we might be tempted to use one for each class independently and then combine them in some way. However this can lead to some serious issues as we show in figure 5.1 below.

DS_IMG156 Fig 5.1

Figure (5.135): similar to Bishop 2006. (Left): two binary classifiers used independently in a one-versus-all (OvA) fashion to decide whether an instance belongs or does not belong to a class. (Right): Binary classifiers used independently in one-versus-one fashion to decide if an instance belongs to one of two classes.

In figure $5.1$ above, the yellow area to the left is an area of ambiguity where the two classifiers can dictate that an
instance belongs to both class $C_{1}$ and class $C_{2}$. The yellow area to the right is an area of ambiguity where an instance can be decided to belong to two or three classes at the same time. The solution is to use a classifier with
multi-linear boundaries $y_{k}(x)$ (similar to when we had multi-output regression) and decide on an instance x belongs to class $C_{k}$ only when its activation function $y_{k}(x)$ is higher than all other classes activations $y_{j}(x) .$ One way to do that is by normalising the regularised scores that we obtain from each independent classifier (on the left) and instead of applying one-versus-all approach we choose the highest score. You will see some examples of the effectiveness of this strategy later.

##Multi-output (multi-class) linear classification models

The structure would be similar to the structure that we have seen for the multi-output linear regression which is shown below for convenience.

DS_IMG157 Fig 5.2

Figure (5.236): Schematic representation of a multi-output independent logistic regression linear classifiers with basis.

The linear classification model with multiple output can be expressed as

$$
\boldsymbol{y}(\mathbf{x}, \mathbf{W})=g\left(\mathbf{W}^{\top} \boldsymbol{\phi}(\mathbf{x})\right)
$$

Figure $5.2$ above shows logistic regression with $K$ multiple outputs. The number of outputs can be associated with multiple classes $K$ where in general $\hat{K} \geq K-1$ with $\hat{K}=K-1$ when the classes are linearly $\underline{\underline{\text { separable }}}$ and their boundaries are parallel. For example we may need only 2 lines to separate 3 classes. When the classes are non-linearly separable, $K$ corresponds to the number of decision boundaries needed to separate the classes. The issue with this structure is that it does not harmonise the outputs and may run into the issues mentioned in the previous section (particularly the issues associated with one-versus-all) because of the independence of the outputs. In such cases, further processing will be needed in order to decide which class the data point is from.
Each classifier $i$ is specialised in one class $i$ and produces an independent probability estimate $p_{i}\left(C_{i} \mid \mathbf{x}\right) .$ One of the simplest approaches is to normalise the scores $p_{i}\left(C_{i} \mid \mathbf{x}\right)$ that were produced by the independent classifier to properly obtain a unified probability distribution $p\left(C_{i} \mid \mathbf{x}\right)=\frac{p_{i}\left(C_{i} \mid \mathbf{x}\right)}{\sum_{i=1}^{K} p_{i}\left(C_{i} \mid \mathbf{x}\right)}$ over the different classes $i$. The final decision on which class the data point is from can be performed by picking the class with the max probability; see section $5.2$ of this paper by Zadrozny and Elkan (2002) for more details. See the following Jupyter notebook for more on classifying iris dataset 3 classes, and on how a one-versus-all approach creates an ambiguous triangular decision region. See also here and here to know more about linear classifiers in sklearn.

The same structure can be produced to the perceptron we only need to change the activation function.

DS_IMG158 Fig 5.3

Figure (5.338): Schematic representation of a multi-output independent perceptron linear classifiers with basis

(top l-r) DS_IMG159, DS_IMG160 Fig 5.4

(bottom l-r) DS_IMG161, DS_IMG162 Fig 5.4

Figures (5.4-5.739): Decision boundaries for multi-output (multi-class) perceptron (left) and logistic regression (right) for the iris dataset. The setosa is linearly separable from the rest, while versicolor and virginica are non-linearly separable. (Top) shows one-versus-all (OvA) which creates ambiguous regions (hyperplanes) shown by the dashed lines. For the perceptron it is struggling to come up with close enough boundaries and thrown off by the outliers in both the versicolor and the virginica. The logistic regression one-versus-all did better in that sense because it is much more resilient towards outliers, but it still has an ambiguous region (the triangle in the middle of the figure) where two classifiers are thinking that the data in the middle belongs to their respective positive class. (Bottom) The shaded decision regions are obtained via the normalisation of the scores given by the independent classifiers in order to overcome the ambiguity issue of the OvA approach that we mentioned in the previous section.

DS_IMG163 Fig 5.5

Figures (5.8-5.940): Decision boundaries for multi-output (multi-class) both for logistic regression on the iris dataset. Left shows a multinomial logistic regression while the right shows normalised multi-output logistic regression. The difference is that in the left we normalise an exponential activation functions for multi-output, on the right we normalise a multiple logistic functions instead of exponential functions. As you can see, multinomial logistic regression deals better with ambiguity but it is still there because essentially we are still dealing with linear models. We need a more complex model to deal with issue such as a neural network that is capable of generating a curved shape boundaries.

Watch a video2 that explains the above concepts.

DS-VID-17

##Multinomial logistic regression: Softmax regression

In the last section we saw how logistic regression has been built originally for binary classification where we expect the output of the activation function to give us one value in the range $[0,1]$. The value represents the probability of the input being from the positive class and we get the probability from the negative class by exploiting that the sum of both must be 1 . In other words, if $p\left(C_{1} \mid \mathbf{x}\right)$ is given by the one output of the logistic regression model then $p\left(C_{0} \mid \mathbf{x}\right)=1-p\left(C_{1} \mid \mathbf{x}\right)$

We also saw how we can adapt multiple of these binary classifiers in order to deal with multi-class problem where we have multiple classes that we need to deal with. Basically, we exploited the multi-output structure that we developed in regression and we extended it to multi binary classification to deal with multi-class cases. We have seen three approaches to harmonise or combine the multi-output into one decision: where we can apply OvA or OvO or combine in other ways. We saw also that OvA and OvO approaches create ambiguous decision regions and we saw that that we can deal OvA ambiguity via normalisation.

In this section we will extend the structure of logistic regression into multi-classes. The technique is called multinomial logistic regression because we are dealing with multinomial target variable. Binomial and multinomial are names that comes from probability theory to deal with variables that takes only two values or multiple values. The dependent variable here is the label which can take one of a multiple categorical values. So essentially it is a fancy name for multi-class problems. So we are extending logistic regression from binomial to multinomial. We will start with binary logistic regression that we covered in a previous section.

Let us assume that we build a multi-output model as shown in figure 5.10:

DS_IMG164 Fig 5.6

Figure (5.1041): 2-output linear classifier with exponential activation function

$$
\begin{array}{l}
y_{1}=e^{\mathbf{w}_{1}^{\top} \boldsymbol{\phi}} \\
y_{2}=e^{\mathbf{w}_{2}^{\top} \boldsymbol{\phi}}
\end{array}
$$

$$
\text { Let us normalise } \quad \bar{y}_{1}=\frac{e^{\mathrm{w}_{1}^{\top} \phi}}{e^{\mathbf{w}_{1}^{\top} \phi}+e^{\mathbf{w}_{2}^{\top} \phi}}=\frac{1}{1+\frac{e^{\mathbf{w}_{2}^{\top} \phi}}{e^{\mathbf{W}_{1}^{\top} \phi}}}=\frac{1}{1+e^{\mathbf{w}_{2}^{\top} \phi-\mathbf{w}_{1}^{\top} \phi}}=\frac{1}{1+e^{\left(\mathbf{w}_{2}^{\top}-\mathbf{w}_{1}^{\top}\right) \phi}}=\frac{1}{1+e^{\left(\mathbf{w}_{2}-\mathbf{w}_{1}\right)^{\top} \phi}}
$$

$$
\text { By defining } \mathbf{w}:=\mathbf{w}_{2}-\mathbf{w}_{1} \text { and substituting we get: }
$$

$$
\bar{y}_{1}=\frac{1}{1+e^{\mathbf{w}}^{\top} \phi}
$$
