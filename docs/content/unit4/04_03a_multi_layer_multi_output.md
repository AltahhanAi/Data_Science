#Non-linear regression via neural networks

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * build the architecture of a two layer or multilayer neural network
    *	address the regression problem in full using feature extraction process that is built into the neural network architecture, negating the need from the designer to select fixed basis functions
    *	understand the role of activation function in the context of a neural network
    *	devise a suitable loss function for a neural network
    *	explain the process of building a suitable learning algorithm for a multi layer neural network that depends on the gradient descent and the error propagated through the network’s layers.

**In the previous lesson we have covered a linear regression model with multiple outputs. In this lesson we extend this idea into models that have multiple layers with different activation function.**

The layers should be defined in a way that do not reduce them into one layer in order to justify the added complexity of the new layers. This is not a strict guideline but it undesirable to have redundant layers that can be otherwise replaced by fewer layers. The reducibility of the layers is tightly connected to the form of the activation function.

An activation function is a function that we pass the output though in order transforms the output into a form that more useful for our model. The activation function can be linear or non-linear. In fact, so far we can say that we have been implicitly using an identity activation function (i.e.) the output stays as is. The non-linearity of the activation function is a powerful tool that can transform an input into an output that has gone through considerable processing. The result of such a model with multiple layers and non-linear and linear activation function for each layer is called a neural network.

##Non-linear regression using multi-layer models

We start with the latest architecture that we have developed in the previous sections and amend it to suit our needs. Please bear in mind that we are building the simplest feedforward neural network here, but there are much more complex networks architecture that you will see later in the machine learning and deep learning modules.  

We will adopt an approach where we will now develop an architecture that will allow us to adapt the basis function that we have assumed previously as being fixed. The number of the features are usually fixed in neural networks because it corresponds with number of neurons in a layers. Other techniques such as support vector machine allow for the flexibility of the adapting the number of feature basis according to the dataset but on the expense of less efficiency during the prediction. In neural networks the adaptation takes place in changing the expressing powers of the basis function by changing the weights of the hidden layer. Below we show how we move from a fixed basis model architecture into flexible adapted basis model architecture.

<mark> figure to be replaced</mark>
<figure role="group">
  <img src="../images/DS_IMG122.png" alt="Schematic representation of a multiple outputs multi-layers linear regression model with fixed basis." />
  <figcaption><strong>Figure 4.25</strong> Schematic representation of a multiple outputs multi-layers linear regression model with fixed basis.</figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG123.png" alt="Schematic representation of a non-linear, multi-layer neural network model with adaptive basis." />
  <figcaption><strong>Figure 4.26</strong> **Non-linear** multi-layer **Neural Network** model with adaptive basis.</figcaption>
</figure>

First, we denote the weight matrix that links the input features with the first set of linear models (called the hidden layer) as $\dot{\mathbf{W}}$ and it is of dimension M×(D+1) where $D$ is the input space dimension and $M$ is the number of features that we would like to obtain from the hidden layer. Next, we denote the weight vector that links the hidden linear models with the output models (called output layer) as $\mathbf{W}$ and it is of dimension $(M+1)$. Note that this architecture cannot be reduced into one-layer output as we did earlier due to the presence of the activation function. Which support what we have mentioned earlier that this the simplest ANN for regression. So, what are the advantages of such an architecture over the one-layer architecture you might ask? The answer is that it allows us to capture more complex relationship between the input and the output automatically without the need to come up with a suitable basis functions.

Both layers can be seen as linear regression model that have been cascaded together. The question remains to study how they interact with each other. The hidden layer output is given as:

$$
\boldsymbol{y}=\mathbf{w}^{\top} \boldsymbol{\Phi}
$$

Where $\phi(\mathbf{x})$ is denoted as $\boldsymbol{\phi}$ and $\boldsymbol{\Phi}$ is a feature vector that includes a dummy feature $ϕ_0=1$. The output of the input layer is given as:

$$
\boldsymbol{\phi}=f\left(\dot{\mathbf{W}}^{\top} \mathbf{x}\right)
$$

Where $\mathbf{X}$ is a vector that dummy attribute $x_{0}=1$ and $f$ is an activation function that can be non-linear such as the sigmoid and the tanh or a linear one such as a step function. An important aspect of the activation function is that it should be differentiable in order for the optimisation to be possible and for learning to take place.

So if we decided to choose a sigmoid activation function then the full network can be expressed as:

$$
\boldsymbol{y}=\mathbf{w}^{\top} \boldsymbol{\Phi} \quad \boldsymbol{\phi}=g\left(\dot{\mathbf{W}}^{\top} \mathbf{x}\right) \mid
$$

This is called a forward propagation which will give us a multi-output prediction for an input $\mathbf{x}$ (both $\mathbf{x}$ and $\boldsymbol{\Phi}$ has a dummy component.

Note that if we put the non-linear activation function on the output units and the linear on the hidden layer then we end up reducing the whole network into a one layer network with non-linear activation function so we effectively lose the hidden layer. More formally, let us ignore the biases for a moment, we would have $\boldsymbol{y}=\mathrm{g}\left(\mathbf{w}^{\top} \dot{\mathbf{W}}^{\top} \mathbf{x}\right)$, now if we define $\ddot{\mathbf{W}}^{\top}=\mathbf{w}^{\top} \dot{\mathbf{W}}^{\top}$ then we can reduce the model into $\boldsymbol{y}=\mathrm{g}\left(\ddot{\mathbf{W}}^{\top} \mathbf{x}\right)$ which is equivalent to a one layer non-linear model.

##The activation function

Note that for the activation function, we have $w^⊤ x$ on the horizontal axis and y on the vertical axis, so please do not mix between $x_2$ and $y$ they are two different things; $x_2$ is an attribute and it participates in forming the depicted decision boundaries, while $y$ is a label. More explicitly, in regression the straight line equations in $2D$ represented the relationship between a one attribute $x$ and the label $y$. On the other hand, the straight line here represents the relationship between attributes $x_1$ and $x_2$ and is used to separate the classes using a step activation function.

If we want to confine the values to a $]0,1[$ interval while allowing the activation function to take values in between to reflect the strength of the belief, or the probability, that a data point $x_n$ belongs (or not) to the positive class then we can use the logistic function shown below.

<mark> figure to be replaced with DS_IMG211</mark>

<figure role="group">
  <img src="../images/DS_IMG124.png" alt="Graph showing a logistic activation function." />
  <figcaption><strong></strong> </figcaption>
</figure>

**<p style="text-align: center;">Figure 4.27:** *Logistic function: (left) logistic function in 2d space with one attribute x. where we can see that the logistic has an inflection point at x=0 where its curvature changes from concave-upward to concave-downward, at this point the logistic value is y=0.5. (right): logistic surface in 3d space with two attributes $x_{1}$ and $x_{2}$, where w=[0.6,0.6]. the surface has an inflection surface at x=0.*</p>

This activation function is used to be the most common activation function for hidden layers in neural networks for treating non-linear models. It is still an important one that create synergy with a different loss function called cross entropy for classification as we shall see later in the next unit. We will use it when we move from one-layer model (including multi-output one) to multi-layer models we need to adjust our cost function.

##Loss functions

Earlier we saw that for regression the mean squared errors is useful loss function. With multi-outputs we did not need to change the cost function because each output acts independently and has its own loss function we do not need to tie them up together because each weight vector lives by its own and do not affect other weight vectors. So for example one of the weights vectors that corresponds to an output can be frozen while we train the other outputs weights and we still get the same results when we allow it to train with the rest of the weights team. The central idea that we are trying to convey here is that the outputs are independent even if the different outputs are related. So, for the one layer multi-output model the cost functions of the output can be written independently as:

$$
\overline{J^{2}}(\mathbf{w})=\frac{1}{2 N}\|\mathbf{t}-\mathbf{y}(\mathbf{X}, \mathbf{w})\|^{2}
$$

$$
\overline{J^{2}}(\mathbf{w})=\frac{1}{2 N}\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2}
$$

$$
\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2}
$$

$$
\nabla \overline{J^{2}}(\mathbf{W})=\frac{1}{N} \mathbf{\Phi}^{\top}(\mathbf{t}-\mathbf{\Phi} \mathbf{w})
$$

One the other hand, when we move to a multi-layer topology similar to the above, things change dramatically. Suddenly, the weights of the hidden layer will affect the performance of the consequent layer (output layer) and even though we can still treat the output weights independently the performance of the hidden layer will directly affect the performance of the whole model. And hence we need now to coordinate the training of all the outputs together because otherwise changing the hidden layer in isolation according to one output will have inadvertent implications on the performance of the other outputs. For example if the hidden layer is trained with one of the output is frozen (or by ignoring the performance of one of the outputs), then the resultant hidden weights will be good for all outputs except for the one that we have not incorporated in our training, and hence the performance of the network may become biased against this particular frozen output.

It is true that we can try to make the output weights compensate for the lack of training, but this is not guaranteed. We can start by a random hidden layer and train only the output layer which is one of the main ideas of models called extreme machines. In this case we would need a large enough hidden layer to give us enough variety and diversity to encode input for the output layer. There are lots of debates around whether we need extreme machines and whether its ideas are novel, in any case it is outside the scope of our coverage.

The loss function cannot be expressed as a series of loss function as the case in the one layer multi-outputs case and must stay in vectorised form to make sure that the hidden unit are optimised according to all of the outputs and not favour one over the other.

$$
\bar{J}(\mathbf{w}, \dot{\mathbf{W}})=\frac{1}{2 N}\left(\|\mathbf{t}-\mathbf{t}(\mathbf{X}, \mathbf{w}, \dot{\mathbf{W}})\|^{2}\right)
$$

$$
\bar{J}(\mathbf{w}, \dot{\mathbf{W}})=\frac{1}{2 N}\left(\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2}\right)
$$

$$
\bar{J}(\mathbf{w}, \dot{\mathbf{W}})=\frac{1}{2 N}\left(\left\|\mathbf{t}-\left[\begin{array}{c}
\mathbf{1} \\
g(\mathbf{X} \dot{\mathbf{W}})
\end{array}\right] \mathbf{w}\right\|^{2}\right)
$$

$$
g(\alpha)=\frac{1}{1+e^{-\alpha}}
$$

Taking the derivative and assuming that we have sigmoid activation function g yields:

$$
\nabla \bar{J}(\mathbf{w})=\frac{1}{N} \mathbf{\Phi}^{\top}(\mathbf{t}-\mathbf{\Phi} \mathbf{w}) \mid
$$

##Multi-layer neural network learning for one output

In this section we generalise the ideas of multi-output regression to a multi-layer neural network with one output. This an essential distinction in order to be able to build later on this architecture for a classification model with multi-class output in unit 5.

<mark>Figure **4.28** schematic representation of the multi-layer perceptron as a non-linear models with one output.</mark>

$$
\bar{J}(\mathbf{w}, \dot{\mathbf{W}})=\frac{1}{2 N}\left(\|\mathbf{t}-g(\mathbf{X} \dot{\mathbf{W}}) \mathbf{w}\|^{2}\right)
$$

For the weights $\dot{\mathbf{W}}$ ̇the gradient can be obtained via the derivative chain rule as:

$$
y=\Phi \mathbf{w}
$$

$$
\nabla \bar{J}\left( \dot{\mathbf{W}}_{: k}\right)=\frac{1}{2 N}\left[\mathbf{X}^{\top}[(\mathbf{\Phi} \circ(\mathbf{1}-\mathbf{\Phi})) \times(\boldsymbol{t}-\boldsymbol{y})]\right] \times \mathbf{w}^{\top}
$$

The gradient takes a slightly more complex form due to the introduction of the nonlinearity on the hidden layer activation function.

The formula can be deduced by realising that $g(\mathbf{X} \dot{\mathbf{W}})=\mathbf{\Phi}$ and its gradient is $\nabla \boldsymbol{g}=\mathbf{\Phi} \circ(\mathbf{1}-\mathbf{\Phi})$, where $\circ$ is element-wise matrix multiplication. The gradients $\nabla \mathbf{X} \dot{\mathbf{W}}=\mathbf{X}^{\top}$ and $\nabla_{\mathbf{\Phi}} \mathbf{\Phi} \mathbf{w}=\mathbf{w}^{\top}$ are at the two edges of the formula. Finally, $\times$ indicates broadcasting operation. Broadcasting allows us to repeatedly multiply elements from one vector by all vectors of a matrix if the matrix and the vector have the same number of elements on one of the dimensions. In this case we have that $(\boldsymbol{t}-\boldsymbol{y})$. We will give you a full code of a project that shows each operation individually. Your search for broadcasting in numpy will be helpful to understand more about this topic. Note that we need to iterate through the outputs components in order to update all columns of $\dot{\mathbf{W}}$.

From here it can be seen that we are unable to do least squares on the cost function and we only can reside to an approximation of the optimal weights $\mathbf{W}^{*}$ and $\mathbf{W}^{*}$.

!!! algorithm-heading "Algorithm 4: Regularised Mini-Batch Stochastic Gradient Descent learning for two layers Neural Network Model with sigmoid and identity activation functions for the hidden and output layers respectively‎"

  	**Input:**

    !!! algorithm ""

    	Input set: design matrix $\mathbf{X}=\left[\mathbf{x}_{1}^{\top}, \ldots, \mathbf{x}_{N}^{\top}\right]^{\top} \operatorname{each} \mathbf{x}_{n}$ is of size $D$ <span class="algorithm-line-comment"> *Training set*</span>

    	Labels set: vector $\mathbf{t}=\left[t_{1}, \ldots, t_{N}\right]^{\top}$ each $\boldsymbol{t}_{n}$ is a scalar.

    	$\mathbf{X}^{\prime}, \mathbf{t}^{\prime}$ holdout validation set that have similar structure to the above with size $v$

    	$\eta_{0}$: initial learning rate

        $b$: mini-batch size (specifies how frequent we want to update the weights $\mathbf{w}$ and $\dot{\mathbf{W}}$ )

    	$\lambda:$ regularisation parameter

    	ep: max number of epochs

    	$\varepsilon:$ early stopping threshold


  	**Output**: $\mathbf{w}$ and $\dot{\mathbf{W}}$ approximations for optimum weights $\mathbf{w}^{*}$ and $\mathbf{W}^{*}$; of size $M+1$ and $(D+1) \times M$ respectively.

  	**NN regression** $\left(\mathbf{X}, \mathbf{t}, \mathbf{X}^{\prime}, \mathbf{t}^{\prime}, \eta_{0}, b, \lambda, e p, \varepsilon\right)$:

    !!! algorithm ""
        Initialise $\mathbf{w}, \dot{\mathbf{W}}$ arbitrarily and assign weights backups $\mathbf{w}^{\prime}=\mathbf{w}$ and $\dot{\mathbf{W}}^{\prime}=\dot{\mathbf{W}}$ as well as $\eta=\eta_{0}$ and $\bar{J}_{0}=\infty$

        For epoch $= 1:ep$
        <span class="algorithm-line-comment"># *hyper parameter: max number of epochs*</span>

        !!! algorithm ""
            For iteration $\tau=1: q$
            <span class="algorithm-line-comment"># *$q≥N/ b$*</span>

            !!! algorithm ""

                Select a mini-batch $\mathbf{X}_{\tau}, \mathbf{T}_{\tau}$ of size $b$ from $\mathbf{X}, \mathbf{T}$
                <span class="algorithm-line-comment"># *randomly or by shuffling & partitioning*</span>

				$\mathbf{X}_{\tau}=\left[\mathbf{1}_{\mathbf{b}}, \mathbf{X}_{\tau}\right]$
                <span class="algorithm-line-comment"># *add dummy feature to the design matrix*</span>

                $\mathbf{\Phi}_{\tau}=g\left(\mathbf{x}_{\tau} \mathbf{w}\right)$
                <span class="algorithm-line-comment"># *element-wise sigmoid $g(\alpha)=\frac{1}{1+e^{-\alpha}}$*</span>

				$\mathbf{W}^{\prime}=\left(1-\frac{1}{b} \eta \lambda\right) \mathbf{W}^{\prime}+\frac{1}{b} \eta \mathbf{\Phi}_{\tau}^{\top}\left(\mathbf{T}_{\tau}-\mathbf{\Phi}_{\tau} \mathbf{W}^{\prime}\right)$
                <span class="algorithm-line-comment"># *update with regularisation*</span>

				$\dot{\mathbf{W}}^{\prime}=\left(1-\frac{1}{b} \eta \lambda\right) \dot{\mathbf{W}}^{\prime}+\frac{1}{b} \eta \mathbf{X}_{\tau}^{\top}\left(\mathbf{T}_{\tau}-\mathbf{\Phi}_{\tau} \mathbf{W}^{\prime}\right) \mathbf{W}^{\prime \top} \mathbf{\Phi}_{\tau}\left(\mathbf{1}-\mathbf{\Phi}_{\tau}\right)$

			Decay $η$
            <span class="algorithm-line-comment"># *if necessary*</span>

			$\mathbf{\Phi}^{\prime}=g\left(\mathbf{X}^{\prime} \mathbf{W}\right)$

			$\bar{J}_{e p}=\frac{1}{2 N}\left\|\mathbf{T}^{\prime}-\mathbf{\Phi}^{\prime} \mathbf{W}^{\prime}\right\|^{2}$
            <span class="algorithm-line-comment"># *calculate the loss or other metric on the validation set*</span>

			If $\bar{J}_{e p}>\bar{J}_{e p-1}+\varepsilon:$ break
            <span class="algorithm-line-comment"># *simple early stopping or other more sophisticate cond.*</span>

			Else $\mathbf{W}=\mathbf{W}^{\prime}$ and $\mathbf{W}=\mathbf{W}^{\prime}$

        Return the final solution $W$ and $\dot{\mathbf{W}}$

Note that we did not have an algorithm for the least squares because we cannot do it for multi-layer non-linear neural network. In all of the algorithms for mini-batch we have given them number 4 (4, 4’, 4’’, 4’’’) the ‘ signifies the stage of the algorithm, where they cover: linear, linear with basis, linear with basis and multi-outputs and non-linear multi-layer neural network, respectively.

!!!info "Preventing overfitting for neural networks"
	To understand how to prevent overfitting in neural networks look at the section **Preventing overfitting the data and overshooting the loss minimum** and at Algorithm 5''.


!!! abstract "Exercise"

    See the following Jupyter notebook for an example of non-linear regression.

    <mark>FILE MISSING</mark>

    <a href="../exercises/xxx.ipynb" target="_blank" download>Non-linear regression neural network Jupyter Notebook (.ipynb)</a>

##Summary

In this section we have covered the basics of neural networks and we took the liberty to simplify it coverage. You will study this topic extensively in machine learning and deep learning. An important aspect of neural networks is that they allow us to represents arbitrary relationship between input and output linear and non-linear. Therefore, they are called universal approximator. In the next unit we will take advantage of the knowledge that you gained in dealing with regression to extend it to numerical classification.

<mark>Watch a video3 that explains the above concepts.</mark>
