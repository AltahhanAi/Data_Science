#Non-linear regression via neural networks

In the previous section we have covered a linear regression model with multiple outputs. In this section we extend this idea into models that have multiple layers with different activation function. The layers should be defined in a way that do not reduce them into one layer in order to justify the added complexity of the new layers. This is not a strict guideline but it undesirable to have redundant layers that can be otherwise replaced by fewer layers. The reducibility of the layers is tightly connected to the form of the activation function.

An activation function is a function that we pass the output though in order transforms the output into a form that more useful for our model. The activation function can be linear or non-linear. In fact, so far we can say that we have been implicitly using an identity activation function (i.e.) the output stays as is. The non-linearity of the activation function is a powerful tool that can transform an input into an output that has gone through considerable processing. The result of such a model with multiple layers and non-linear and linear activation function for each layer is called a neural network.

##Multi-layer multi-output non-linear regression models

We start with the latest architecture that we have developed in the previous sections and amend it to suit our needs. Please bear in mind that we are building the simplest feedforward neural network here, but there are much more complex networks architecture that you will see later in the machine learning and deep learning modules.  

We will adopt an approach where we will now develop an architecture that will allow us to adapt the basis function that we have assumed previously as being fixed. The number of the features are usually fixed in neural networks because it corresponds with number of neurons in a layers. Other techniques such as support vector machine allow for the flexibility of the adapting the number of feature basis according to the dataset but on the expense of less efficiency during the prediction. In neural networks the adaptation takes place in changing the expressing powers of the basis function by changing the weights of the hidden layer. Below we show how we move from a fixed basis model architecture into flexible adapted basis model architecture.

<figure role="group">
  <img src="../images/DS_IMG122.png" alt="Schematic representation of a multiple outputs multi-layers linear regression model with fixed basis." />
  <figcaption><strong>Figure 4.25</strong> Schematic representation of a multiple outputs multi-layers linear regression model with fixed basis.</figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG123.png" alt="Schematic representation of a non-linear, multi-layer neural network model with adaptive basis." />
  <figcaption><strong>Figure 4.26</strong> **Non-linear** multi-layer **Neural Network** model with adaptive basis.</figcaption>
</figure>

!!!info "Important"

	Below we derive equations for a slightly different network without the bias neuron of the feature layer to make the equations easier to handle. $\mathbf{w}_{i}^{\top}$ is row $i$ of the weight matrix $W^⊤; i=1,..K$. Similarly, $\dot{\mathbf{w}}_{j}^{\top}$ is the row $\mathrm{j}$ of the weight matrix $\mathbf{\mathbf { W }}^{\top} ; \mathrm{j}=1_{\ldots .} . \mathrm{M}$.

First, we denote the weight matrix that links the input features with the first set of linear models (called the hidden layer) as $\dot{\mathbf{W}}$ and it is of dimension M×(D+1) where $D$ is the input space dimension and $M$ is the number of features that we would like to obtain from the hidden layer. Next, we denote the weight matrix that links the hidden linear models with the output models (called output layer) as $W$ and it is of dimension $K×(H+1)$. In our depicted example we have the number of hidden neurons on the hidden layer is 3 so $H=4$ and the number of outputs is 2 so $K=2$. Note that this architecture cannot be reduced into one-layer output as we did earlier due to the presence of the activation function. Which support what we have mentioned earlier that this the simplest ANN for regression. So, what are the advantages of such an architecture over the one-layer architecture you might ask? The answer is that it allows us to capture more complex relationship between the input and the output. So, for example such an architecture will be able to capture and model the following data/relationship:

Both layers can be seen as multi-output linear regression model. So, let us see how they interact with each other: the hidden layer output is given as:

$y=W^⊤ ϕ$

Where $ϕ(x)$ is denoted as ϕ and is a feature vector that includes a dummy feature $ϕ_0=1$. The output of the output layer is given as: $\mathbf{x}=\left[\begin{array}{l}1 \\ \boldsymbol{x}\end{array}\right]$ and $\boldsymbol{\phi}=\left[\begin{array}{l}1 \\ \boldsymbol{\phi}\end{array}\right]$

$ϕ=g(W ̇^⊤ x)$

Where $f$ is an activation function that can be non-linear such as the sigmoid and the tanh or a linear one such as a step function. An important aspect of the activation function is that it should be differentiable in order for the optimisation to be possible and for learning to take place.

So the full network can be expressed as:

$\boldsymbol{y}=\mathbf{W}^{\top}\left[\begin{array}{c}1 \\ g\left(\mathbf{\mathbf { W }}^{\top} \mathbf{x}\right)\end{array}\right]$

This is called a forward propagation which will give us a multi-output prediction for an input $x$.

Let us ignore the biases for a moment in order to simplify what is entailed in this simple neural network. Note that if we put the non-linear activation function on the output units and the linear on the hidden layer then we end up reducing the whole network into a multi-output one layer with non-linear activation function so we effectively loose the hidden layer. $\boldsymbol{y}=\mathrm{g}\left(\mathbf{W}^{\top} \mathbf{W}^{\top} \mathbf{x}\right)$ if we define $\ddot{\mathbf{W}}^{\top}=\mathbf{W}^{\top} \mathbf{\mathbf { W }}^{\top}$ then we can reduce the model into $\boldsymbol{y}=\mathrm{g}\left(\ddot{\mathbf{W}}^{\top} \mathbf{x}\right)$ which is equivalent to a one layer multi-output non-linear model.

##The activation function

It is important to note that we need to use a function $f$ to map the values $w^⊤ x$ into some sort of decision that is related to the label.

There are some important and often overlooked subtleties to notice here. First $y$ is not the boundaries $w^⊤ x$ although it is inferred from it via $f$. Secondly, $y$ is fundamentally not a continuous value unlike the attribute $x2$ for example. Nevertheless, it can be mapped into a continuous space to follow the continuum of values $w^⊤ x$ can take as we saw earlier in the step activation function where essentially we have all positive $w^⊤ x$ values are mapped into $y=1$ and all negative values of $w^⊤ x$ are mapped into the value $-1$. So any time the value of $w^⊤ x_n$ is positive for some data point $x_n$ then we classify $x_n$ as belonging to class $+1$ and conversely any time we have that $w^⊤ x_n$ is negative then we classify $x_n$ to be of class $-1$.

Note that for the activation function, we have $w^⊤ x$ on the horizontal axis and y on the vertical axis, so please do not mix between $x_2$ and $y$ they are two different things; $x_2$ is an attribute and it participates in forming the depicted decision boundaries, while $y$ is a label. More explicitly, in regression the straight line equations in $2D$ represented the relationship between a one attribute $x$ and the label $y$. On the other hand, the straight line here represents the relationship between attributes $x_1$ and $x_2$ and is used to separate the classes using a step activation function.

If we want to confine the values to a $]0,1[$ interval while allowing the activation function to take values in between to reflect the strength of the belief, or the probability, that a data point $x_n$ belongs (or not) to the positive class then we can use the logistic function shown below.

<figure role="group">
  <img src="../images/DS_IMG124.png" alt="Graph showing a logistic activation function." />
  <figcaption><strong>Figure 4.27</strong> </figcaption>
</figure>

This activation function is used to be the most common activation function for hidden layers in neural networks for treating non-linear models. It is still an important one that create synergy with a different loss function called cross entropy for classification as we shall see later in the next unit. We will use it when we move from one-layer model (including multi-output one) to multi-layer models we need to adjust our cost function.

##Loss functions

Earlier we saw that for regression the mean squared errors is useful loss function. With multi-outputs we did not need to change the cost function because each output acts independently and has its own loss function we do not need to tie them up together because each weight vector lives by its own and do not affect other weight vectors. So for example one of the weights vectors that corresponds to an output can be frozen while we train the other outputs weights and we still get the same results when we allow it to train with the rest of the weights team. The central idea that we are trying to convey here is that the outputs are independent even if the different outputs are related. So, for the one layer multi-output model the cost functions of the output can be written independently as:

$\begin{array}{c}
\bar{J}(\mathbf{W})=\frac{1}{2 N}\|\mathbf{T}-\mathbf{Y}(\mathbf{X}, \mathbf{W})\|^{2} \\
\bar{J}(\mathbf{W})=\frac{1}{2 N}\|\mathbf{T}-\mathbf{\Phi} \mathbf{W}\|^{2}
\end{array}$

Which equivalently can be written as a series of objective functions each correspond to one output as follows:
$\bar{J}_{k}\left(\mathbf{w}_{k}\right)=\frac{1}{2 N}\left\|\mathfrak{t}_{k}-\mathbf{\Phi} \mathbf{w}_{k}\right\|^{2}$

Where $\mathfrak{t}_{k}$ is a vector of size $N$ the corresponds to outputs of the entire dataset for only one component $k$ which is a column in matrix $\mathbf{T}$ (we have denoted as such to differentiate it form $\mathbf{t}_{n}$ that corresponds with target vector of all outputs of data point $\mathbf{x}_{n}$ )

$\begin{array}{c}
\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2} \\
\nabla \bar{J}(\mathbf{W})=\frac{1}{N} \mathbf{\Phi}^{\top}(\mathbf{T}-\mathbf{\Phi W})
\end{array}$

One the other hand, when we move to a multi-layer topology similar to the above, things change dramatically. Suddenly, the weights of the hidden layer will affect the performance of the consequent layer (output layer) and even though we can still treat the output weights independently the performance of the hidden layer will directly affect the performance of the whole model. And hence we need now to coordinate the training of all the outputs together because otherwise changing the hidden layer in isolation according to one output will have inadvertent implications on the performance of the other outputs. For example if the hidden layer is trained with one of the output is frozen (or by ignoring the performance of one of the outputs), then the resultant hidden weights will be good for all outputs except for the one that we have not incorporated in our training, and hence the performance of the network may become biased against this particular frozen output.

It is true that we can try to make the output weights compensate for the lack of training, but this is not guaranteed. We can start by a random hidden layer and train only the output layer which is one of the main ideas of models called extreme machines. In this case we would need a large enough hidden layer to give us enough variety and diversity to encode input for the output layer. There are lots of debates around whether we need extreme machines and whether its ideas are novel, in any case it is outside the scope of our coverage.

The loss function cannot be expressed as a series of loss function as the case in the one layer multi-outputs case and must stay in vectorised form to make sure that the hidden unit are optimised according to all of the outputs and not favour one over the other.

$\bar{J}(\mathbf{W}, \mathbf{w})=\frac{1}{2 N}\left(\|\mathbf{T}-\mathbf{Y}(\mathbf{X}, \mathbf{W}, \dot{\mathbf{W}})\|^{2}\right)$

$\bar{J}(\mathbf{W}, \mathbf{w})=\frac{1}{2 N}\left(\|\mathbf{T}-\mathbf{\Phi} \mathbf{W}\|^{2}\right)$

$\bar{J}(\mathbf{W}, \mathbf{w})=\frac{1}{2 N}\left(\left\|\mathbf{T}-\left[\begin{array}{c}\mathbf{1} \\ g(\mathbf{X} \mathbf{w})\end{array}\right] \mathbf{w}\right\|^{2}\right)$

$g(\alpha)=\frac{1}{1+e^{-\alpha}}$

Taking the derivative and assuming that we have sigmoid activation function g yields:

$\nabla \bar{J}(\mathbf{W})=\frac{1}{N} \mathbf{\Phi}^{\top}(\mathbf{T}-\mathbf{\Phi} \mathbf{W})$

!!!info "Important"
	Below we state the gradient and the equations for a slightly different network than the one in the figures above: it is without the bias neuron of the feature layer to make the equations easier to handle.

$\bar{J}(\mathbf{w}, \mathbf{w})=\frac{1}{2 N}\left(\|\mathbf{T}-g(\mathbf{x} \dot{\mathbf{W}}) \mathbf{w}\|^{2}\right)$

For the weights $\dot{\mathbf{W}}$ ̇the gradient can be given via the derivative chain rule as:

$\nabla \bar{J}(\mathbf{w})=\frac{1}{2 N} \mathbf{X}^{\top}(\mathbf{T}-\mathbf{\Phi} \mathbf{W}) \mathbf{W}^{\top} \mathbf{\Phi}(\mathbf{1}-\mathbf{\Phi})$

This is because $g(\mathbf{x} \dot{\mathbf{W}})=\mathbf{\Phi}$ and its gradient is $\nabla g=\mathbf{\Phi}(\mathbf{1}-\mathbf{\Phi})$, while the of $\nabla \mathbf{X} \dot{\mathbf{W}}=\mathbf{X}^{\top}$ and $\nabla_{\Phi} \mathbf{\Phi W}=\mathbf{W}^{\top}$.

From here it can be seen that we are unable to do least squares on the cost function and we only can reside to an approximation of the optimal weights $\mathbf{W}^{*}$ and $\mathbf{W}^{*}$.

The update rules can be written as:

!!! info "Algorithm 4''': Regularised Mini-Batch Stochastic Gradient Descent learning for two layers Neural Network Model with sigmoid and identity activation functions for the hidden and output layers respectively‎"

  	**Input:**

		Input set as a design matrix $\mathbf{X}=\left[\mathbf{x}_{1}^{\top}, \ldots, \mathbf{x}_{N}^{\top}\right]^{\top} \operatorname{each} \mathbf{x}_{n}$ is of size $D$

		Labels set as a matrix $\mathbf{T}=\left[\boldsymbol{t}_{1}^{\top}, \ldots, \boldsymbol{t}_{N}^{\top}\right]^{\top}$ each $\boldsymbol{t}_{n}$ is of size $K$

		$\mathbf{X}^{\prime}, \mathbf{T}^{\prime}$ holdout validation set that have similar structure to the above $\eta_{0}$ : initial learning rate

		b: mini-batch size (specifies how frequent we want to update the weights $\mathbf{W}$ and $\mathbf{W}$ )

		$\lambda:$ regularisation parameter

		ep: max number of epochs

		$\varepsilon:$ early stopping threshold

  	**Output**: $\mathbf{W}$ and $\mathbf{W}$ an approximation for optimum weights $\mathbf{W}^{*}$ and $\mathbf{W}^{*}$; a matrix of size $M \times K$ and $(D+1) \times M$ respectively.

  	**RegSGD VMiniBLRegressNN** (X, $\left.\mathbf{T}, \mathbf{X}^{\prime}, \mathbf{T}^{\prime}, \eta_{0}, b, \lambda, e p, \varepsilon\right)$:

    !!! quote ""
        initialise $\mathbf{W}, \mathbf{w}, \mathbf{W}^{\prime}=\mathbf{W}, \dot{\mathbf{W}}^{\prime}=\mathbf{\mathbf { W }}, \eta=\eta_{0} \text { and } \bar{J}_{0}=\infty$

        For epoch $= 1:ep$ <span style="float: right;"># hyper parameter: max number of epochs</span>

        !!! quote ""
            For iteration $\tau=1: q$ <span style="float: right;"># $q≥N/ b$</span>

            !!! quote ""

                Select a mini-batch $\mathbf{X}_{\tau}, \mathbf{T}_{\tau}$ of size $b$ from $\mathbf{X}, \mathbf{T}$ <span style="float: right;"># randomly or by shuffling & partitioning</span>

								$\mathbf{X}_{\tau}=\left[\mathbf{1}_{\mathbf{b}}, \mathbf{X}_{\tau}\right]$ <span style="float: right;"># add dummy feature to the design matrix</span>

                $\mathbf{\Phi}_{\tau}=g\left(\mathbf{x}_{\tau} \mathbf{w}\right)$ <span style="float: right;"># element-wise sigmoid $g(\alpha)=\frac{1}{1+e^{-\alpha}}$</span>

								$\mathbf{W}^{\prime}=\left(1-\frac{1}{b} \eta \lambda\right) \mathbf{W}^{\prime}+\frac{1}{b} \eta \mathbf{\Phi}_{\tau}^{\top}\left(\mathbf{T}_{\tau}-\mathbf{\Phi}_{\tau} \mathbf{W}^{\prime}\right)$ <span style="float: right;"># update with regularisation</span>

								$\dot{\mathbf{W}}^{\prime}=\left(1-\frac{1}{b} \eta \lambda\right) \dot{\mathbf{W}}^{\prime}+\frac{1}{b} \eta \mathbf{X}_{\tau}^{\top}\left(\mathbf{T}_{\tau}-\mathbf{\Phi}_{\tau} \mathbf{W}^{\prime}\right) \mathbf{W}^{\prime \top} \mathbf{\Phi}_{\tau}\left(\mathbf{1}-\mathbf{\Phi}_{\tau}\right)$

								Decay $η$ <span style="float: right;"># if necessary</span>

								$\mathbf{\Phi}^{\prime}=g\left(\mathbf{X}^{\prime} \mathbf{W}\right)$

								$\bar{J}_{e p}=\frac{1}{2 N}\left\|\mathbf{T}^{\prime}-\mathbf{\Phi}^{\prime} \mathbf{W}^{\prime}\right\|^{2}$ <span style="float: right;"># calculate the loss or other metric on the validation set</span>

								If $\bar{J}_{e p}>\bar{J}_{e p-1}+\varepsilon:$ break <span style="float: right;"># simple early stopping or other more sophisticate cond.</span>

								Else $\mathbf{W}=\mathbf{W}^{\prime}$ and $\mathbf{W}=\mathbf{W}^{\prime}$

        Return the final solution $W$ and $\dot{\mathbf{W}}$

Note that we did not have an algorithm for the least squares because we cannot do it for multi-layer non-linear neural network. In all of the algorithms for mini-batch we have given them number 4 (4, 4’, 4’’, 4’’’) the ‘ signifies the stage of the algorithm, where they cover: linear, linear with basis, linear with basis and multi-outputs and non-linear multi-layer neural network, respectively.

!!!info "Preventing overfitting for neural networks"
	To understand how to prevent overfitting in neural networks look at the section **Preventing overfitting the data and overshooting the loss minimum** and at Algorithm 5''.

##Exercise

See the following Jupyter notebook for an example of non-linear regression.

<a href="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscsaalt%5Fleeds%5Fac%5Fuk%2FDocuments%2FDownloads%2FResources%20for%20ODL%20MSc%2FData%20Science%20Contents%2Funit3%2Fcode%2Fnon%5Flinear%5Fregression%5Fneural%5Fnetwork%2Eipynb&parent=%2Fpersonal%2Fscsaalt%5Fleeds%5Fac%5Fuk%2FDocuments%2FDownloads%2FResources%20for%20ODL%20MSc%2FData%20Science%20Contents%2Funit3%2Fcode&originalPath=aHR0cHM6Ly9sZWVkczM2NS1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC9zY3NhYWx0X2xlZWRzX2FjX3VrL0VjbnFGWURfNjFsSG5sQTBZX2VuY1RNQjRxR0lTMzV1bGNFTnRFZnpDakV0LWc%5FcnRpbWU9OGhPREpoOEwyVWc" target="_blank" class="md-button">Non-linear regression neural network Jupyter Notebook</a>

##Summary

In this section we have covered the basics of neural networks and we took the liberty to simplify it coverage. You will study this topic extensively in machine learning and deep learning. An important aspect of neural networks is that they allow us to represents arbitrary relationship between input and output linear and non-linear. Therefore, they are called universal approximator. In the next unit we will take advantage of the knowledge that you gained in dealing with regression to extend it to numerical classification.

<mark>Watch a video3 that explains the above concepts.</mark>
