#Non-linear models: Multi-layer perceptron

By now you might be thinking: Ok that was all about a straight line-easy! But how about when the classes are non-linearly separable? In this case, we may need more than one line to separate the data or we might need curvy shaped boundaries to separate the classes.  This is where multilayer perceptron comes in handy. But before we rush into this subject, bear in mind that similar to what we have said on linear regression with feature basis, all linear classification models such as perceptron and the logistic regression and the softmax regression can be combined with a feature space (as shown in the algorithms) and hence non-linearly separable classes can become linearly separable classes in a higher dimensional space. However, there are cases where this will not work and we need to move into more flexible features space, where best features are learned, instead of being fixed by the model designer. This will give us a lot of power in expressing an arbitrary decisions boundary, and this is the subject of this lesson.

When rectilinear or multiple lines boundaries are needed to separate the classes. then we can employ multiple independent perceptrons but we will run into the issues that we mentioned earlier in previous section. It will be more convenient, however if we can actually combine these perceptrons in one model that learns the overall best settings of these perceptron together as one comprehensive model and to harmonise and learn the set of parameters needed to identify these set of lines. Furthermore, the possibility of generating more elastic and curvy shaped boundaries is desirable.

The multilayer perceptron does exactly this. It allows us to connect multiple layers of linear and non-linear models together (these can be viewed as computational units that we call neurons–the circles- that we have seen for the perceptron and for the logistic regression). It should be noted however, that the name multilayer perceptron is a misnomer since what we are using in each layer is a logistic regression rather than a perceptron since we will use an activation function for each neuron. Nevertheless, the name is common and we will continue to use it. The key differences between the perceptron and logistics regression is in the activation function treatment, and as long as we are aware of that it should be fine.

Earlier in the regression unit we saw how to build a multi-layer model with activation function on the hidden unit, and how to conduct learning via the backpropagation on it. In this unit we will expand the model that we develop to have a non-linear activation function on the output layer as well. In particular, we use the softmax regression model due to its excellent ability of classifying multiple class problem.

Below we show a generalisation of the concept of moving from one multinomial logistic regression into multi-layer neural networks classifier.

DS_IMG167 Fig 6.1

<figure role="group">
  <img src="../images/DS_IMG167.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 2.1: xxx.</strong></p>
  </figcaption>
</figure>

Figure (6.1): (Top) Schematic representation of a multiple class one-layer logistic regression model (which is considered linear) with fixed basis. (Bottom) Non-linear Multi-layer Multi-class Neural Network model with adaptive basis. We have a softmax activation function for the output and a sigmoid activation function for the hidden layer. The fine dotted box signifies that we normalise the output. The dashed box signifies scaling of the input set.

Backpropagation propagates the error back through the network layers to adjust their weights in a backward manner. We saw how this worked for Algorithm 6''' in the regression unit. So, we need to know first how to adjust the final output to be favourable to our data and classes layout, and then we need to slowly adjust the lines so that this overall performance or ability to distinguish between the classes is increased slowly until it is optimised. This is the idea of stochastic gradient descent with backpropagation.

If we decide to run through all of our data first and formulate a total sum of adjustments that we will execute in one step, then we are talking about batch backpropagation algorithm. Regardless of how we train, the idea is simple: we generalise the concept of one perceptron into multiple ones that act together in harmony. The multi-layer perceptron is also called feedforward neural networks. In fact it is the most common neural networks architecture, but there are plenty of other architectures that are possible. Some are recurrent neural networks: when we allow the network’s past output to participate also as a current input. You will study more about this fascinating topic in the Machine Learning and Deep Learning modules where you will employ automatic differentiation procedure instead of analytically reaching the gradient of the deeply hidden layers. Nevertheless, the complexity of obtaining the gradients for the hidden layers acts as a bridge to appreciate why we need automatic differentiation. It is sufficient to understand that fundamentally we need to take the derivatives and apply the chain rule to propagate the error back in the network.

Watch a video3 that explains the above concepts

##Binary classification using neural network

Let us start with a simpler neural network where we have only one output, i.e. a binary class problem.

The network that we will utilise is expressed as follows.

$$
\begin{array}{l}
y=g\left(\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\right) \\
\boldsymbol{\phi}=g\left(\dot{\mathbf{W}}^{\top} \mathbf{x}\right) \\
y=g\left(\mathbf{w}^{\mathrm{T}} g\left(\dot{\mathbf{W}}^{\top} \mathbf{x}\right)\right)
\end{array}
$$

where we used logits function on both units, the hidden and the output. Its architecture is shown in the figure below and its algorithm is given in the box below. The main difference between this and the regression one is that the classification network uses a non-linear function on the output while for the regression we used a linear activation function on the output. These are not hard rules, but are general enough to be used widely.

<figure role="group">
  <img src="../images/DS_IMG168.png" alt="Brief description." />
  <figcaption>
    <p><strong>Figure 2.1: xxx.</strong></p>
  </figcaption>
</figure> Fig 6.2

Figure (6.254): schematic representation of the multi-layer perceptron as a non-linear models with one outputs.
Essentially the above architecture is an extension of the logistic regression model. We now have a hidden layer that learns the best features representation instead of it being decided by the model designer.

Our loss function is as in the logistic regression case is the cross entropy and is given as:

$$
\begin{array}{c}
\widetilde{H}_{n}(\mathbf{w})=-t_{n} \log \left(y_{n}\right)-\left(1-t_{n}\right) \log \left(1-y_{n}\right) \\
y_{n}=g\left(\mathbf{w}^{\mathrm{T}} g\left(\mathbf{\mathbf { W }}^{\top} \mathbf{x}_{n}\right)\right)
\end{array}
$$

We define

$$
z_{n}=\mathbf{w}^{\mathrm{T}} g\left(\mathbf{\mathbf { W }}^{\top} \mathbf{x}_{n}\right)
$$

We have two derivatives $\nabla_{\mathbf{w}}$ and $\nabla_{\dot{\mathbf{W}}}$ one with respect to each layer weights, for simplicity we will omit the subscript and suffice by showing the loss as a function of the weights that we are differentiating with respect to. The derivative with respect to the output layer is identical to what we saw earlier for the logistic regression model and given as follows:

$$
\nabla \widetilde{H}_{n}(\mathbf{w})=-\boldsymbol{\phi}_{n}\left(y_{n}-t_{n}\right)
$$

The main difference between the algorithms is in the derivation of the hidden units as we saw earlier. It can be proven that when we use logistic function as the activation function, the derivative of the loss is:

$$
\nabla \widetilde{H}_{n}(\dot{\mathbf{W}})=-\nabla z_{n}\left(t_{n}-y_{n}\right)
$$

$$
\begin{array}{l}
\text { And it can be proven }\\
\text { that } \nabla z_{n}=-\mathbf{x}_{n}\left(\mathbf{w} \circ \boldsymbol{\phi}_{n} \circ\left(\mathbf{1}-\boldsymbol{\phi}_{n}\right)\right)^{\top} \text { where } \circ \text { denotes element wise multiplication. }
\end{array}
$$

So we plug now

$$
\nabla \widetilde{H}_{n}(\dot{\mathbf{w}})=-\mathbf{x}_{n}\left(\mathbf{w} \circ \boldsymbol{\phi}_{n} \circ\left(\mathbf{1}-\boldsymbol{\phi}_{n}\right)\right)^{\top}\left(t_{n}-y_{n}\right)
$$

Therefore the final vectorised update takes the form:

$$
\mathbf{w}=\mathbf{w}+\eta\left[\mathbf{X}^{\top}[(\mathbf{\Phi} \circ(\mathbf{1}-\mathbf{\Phi})) \times(\boldsymbol{t}-\boldsymbol{y})]\right] \times \mathbf{w}^{\top}
$$

$$
\text { where } \times \text { denotes broadcasting (a special case of Hadamard product), }
$$

and now we are ready to show the algorithm:

<mark>Algorithm 5</mark>

!!! abstract "Exercise"
    Please the following Jupyter notebook for detail implementation of the 2 layers logistic regression neural networks (same network as above).

      - Download exercise (.ipynb): <a href="" download>xxx</a>

!!! abstract "Exercise"
     See the following Jupyter notebook for a comparison of the regularisation on neural networks.

      - Download exercise (.ipynb): <a href="" download>xxx</a>

!!! abstract "Activity"
    Please the following Jupyter notebook for comparison between different techniques for classification which raps up most of material covered for classification in both unit 2 and unit 3.

     - Download exercise (.ipynb): <a href="" download>xxx</a>

!!! abstract "Activity"
    See the following Jupyter notebook for an out-of-core classification of text documents.

    - Download exercise (.ipynb): <a href="" download>xxx</a>
