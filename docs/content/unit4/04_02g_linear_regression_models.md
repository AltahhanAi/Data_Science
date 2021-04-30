#Linear regression Models

##Regularised Mini-Batch Stochastic Gradient Descent Updates for Linear Regression Model

Similar to what we have done before, there is a regularised version of the minim-batch stochastic gradient descent that we show below:

The regularised loss function (33) for one-output regression problem can be written as:

<mark>$\bar{J}(\mathbf{w})=\frac{1}{2 N}\left(\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2}+\lambda\|\mathbf{w}\|^{2}\right)$</mark>

Therefore, since we have that:

<mark>$\bar{J}(\mathbf{w})=\frac{1}{2 N} \sum_{n=1}^{N} J_{n}^{2}$ then $J_{n}^{2}=\left(t_{n}-\mathbf{w}^{\top} \boldsymbol{\phi}_{n}\right)^{2}+\lambda\|\mathbf{w}\|^{2}$</mark>

This in turn allows us to take the derivative with respect to one data point:

<mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta \frac{1}{2 N} \nabla J_{n}^{2}$</mark>

<mark>$\nabla J_{n}^{2}=-2 \boldsymbol{\phi}_{n}\left(t_{n}-\mathbf{w}^{\top} \boldsymbol{\phi}_{n}\right)+\lambda \mathbf{w}$</mark>

And when we do not know N in advance we just can select a step:

<mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta \frac{1}{N}\left[-\boldsymbol{\phi}_{n}\left(t_{n}-\mathbf{w}^{(\tau)^{\top}} \boldsymbol{\phi}_{n}\right)+\lambda \mathbf{w}^{(\tau)}\right]$</mark>

<mark>$\mathbf{w}^{(\tau+1)}=\left(1-\frac{1}{N} \eta \lambda\right) \mathbf{w}^{(\tau)}+\eta \frac{1}{N} \boldsymbol{\phi}_{n}\left(t_{n}-\mathbf{w}^{(\tau)^{\top}} \boldsymbol{\phi}_{n}\right)$</mark>

The resultant algorithm is similar to Algorithm 4’ and is not shown for brevity. New results on the regularisation can be found in both this paper by Smith et al <a href="https://arxiv.org/pdf/1609.04747.pdf" target="_blank">On the origin of implicit regularization in stochastic gradient descent</a> and this paper by Wei Xu <a href="https://arxiv.org/pdf/2101.12176.pdf" target="_blank">Towards optimal one pass large scale learning with averaged stochastic gradient descent</a>.

##General Case: Linear Models with Multiple Outputs and Fixed Basis

In this section we will extends the ideas of a one output linear regression model that we have dealt with so far into a multi-output linear regression model. When we have multiple output for each input, i.e. each output is a vector of $K$ values: $\boldsymbol{t}_{n}=\left[t_{n, 1}, t_{n, 2}, \ldots, t_{n, K}\right]$. The linear model with multiple output can be expressed as:

(37) <mark>$\boldsymbol{y}(\mathbf{x}, \mathbf{W})=\mathbf{W}^{\top} \boldsymbol{\phi}(\mathbf{x})$</mark>

<figure role="group">
  <img src="../images/DS_IMG009.png" alt="Test image." />
  <figcaption><strong>Figure 4.22</strong> Schematic representation of a **linear regression model** with multiple outputs and fixed basis functions. </figcaption>
</figure>

By fixed basis we mean that the basis functions do not change so if we use a Gaussian basis for example the mean and the variance are fixed, similarly if we use any other basis their parameters do not change. So we can for example, use a separate model that first learns a basis representations (in a separate pre-processing stage) that suits our problem and then we stop the basis learning to make the basis model fixed and then use this fixed basis model to map the input space into our features space and then use the features to learn a multi-output linear model. The resultant model is still linear in the feature space.

Note that we use now a bold face letter $t_n$ to express the fact that we have a vector of multi-output target values. Similarly, we use a bold face $y$ to denote that the model outputs a vector of multi-output values. In addition, we have used a bold capital $W$ to signify that we are dealing with a $(M+1)×K$ matrix (including the biases) instead of a vector of weights. We need in this case a matrix of weights (instead of a vector of weights) since each output component $y_i$ will require its own weights vector. We can combine all the weight vectors in a matrix of weights and we use the same matrix multiplication mechanism that we use before.

We can easily adjust our least square algorithm to accommodate for such a scenario. Here is how:

Staring with the output matrix which we denote now as $T$ is defined as:

<mark>$\mathbf{\Phi}=\left[\begin{array}{c}\boldsymbol{\phi}_{1}^{\top} \\ \vdots \\ \boldsymbol{\phi}_{n}^{\top} \\ \vdots \\ \boldsymbol{\phi}_{N}^{\top}\end{array}\right]=\left[\begin{array}{c}\boldsymbol{\phi}\left(\mathbf{x}_{1}\right)^{\top} \\ \vdots \\ \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\top} \\ \vdots \\ \boldsymbol{\phi}\left(\mathbf{x}_{N}\right)^{\top}\end{array}\right]=\left[\begin{array}{cccc}1 & \phi_{1}\left(\mathbf{x}_{1}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{1}\right) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & \phi_{1}\left(\mathbf{x}_{n}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{n}\right) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & \phi_{1}\left(\mathbf{x}_{N}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{N}\right)\end{array}\right] \mathbf{T}=\left[\begin{array}{c}\boldsymbol{t}_{1}^{\top} \\ \vdots \\ \boldsymbol{t}_{n}^{\top} \\ \vdots \\ \boldsymbol{t}_{N}^{\top}\end{array}\right]=\left[\begin{array}{c}t_{1,1}, t_{1,2} \ldots t_{1, K} \\ \vdots \\ t_{n, 1}, t_{n, 2} \ldots t_{n, K} \\ \vdots \\ t_{N, 1} t_{N, 2} \ldots t_{N, K}\end{array}\right]$</mark>

<mark>$\mathbf{W}^{\top}=\left[\begin{array}{c}\mathbf{w}_{1}^{\top} \\ \vdots \\ \mathbf{w}_{k}^{\top} \\ \vdots \\ \mathbf{w}_{\mathrm{K}}^{\top}\end{array}\right]=\left[\begin{array}{c}w_{0,1}, w_{1,1} \ldots w_{M, 1} \\ \vdots \\ w_{0, k}, w_{1, k} \ldots w_{M, k} \\ \vdots \\ w_{0, K}, w_{1, K} \ldots w_{M, K}\end{array}\right]$</mark>

The model now can be expressed in a full vectorised from as:

(39) <mark>$Y(X,W)=ΦW$</mark>

##Batch Learning: The Least Squares for Multi-Outputs Linear Regression Models with Basis

The loss function for multiple output linear models is defined as:

(40) <mark>$\bar{J}(\mathbf{W})=\frac{1}{2 N} \sum_{n=1}^{N}\left\|\boldsymbol{t}_{n}-\mathbf{W}^{\top} \boldsymbol{\phi}_{n}\right\|^{2}$</mark>

Note here that we are using the norm $‖.‖^2$ since we have a vector of target values. Each operation $t_n-W^⊤ ϕ_n$ produce a vector of errors of size $K$ that corresponds to $K$ different outputs. We take the gradient as usual and set it to 0:

<mark>$\nabla \bar{J}(\mathbf{W})=-\frac{2}{2 N} \sum_{n=1}^{N} \boldsymbol{\phi}_{n}\left(\boldsymbol{t}_{n}-\mathbf{W}^{\top} \boldsymbol{\phi}_{n}\right)^{\top}=0$</mark>

<mark>$\underbrace{\left(\sum_{n=1}^{N} \boldsymbol{\phi}_{n} \boldsymbol{\phi}_{n}^{\top}\right)}_{\boldsymbol{\Phi}^{\top} \boldsymbol{\Phi}} \mathbf{W}^{*}=\underbrace{\sum_{n=1}^{N} \boldsymbol{\phi}_{n} \boldsymbol{t}_{n}^{\top}}_{\boldsymbol{\Phi}^{\top} \mathbf{T}}$</mark>

Where $\boldsymbol{t}_{n}^{\top}=\left[\begin{array}{llll}t_{n, 1} & t_{n, 2} & \ldots & t_{n, K}\end{array}\right]$. Each $\boldsymbol{\phi}_{n} \boldsymbol{t}_{n}^{\top}$ produce a matrix of size $M×D$. We can summarise the operation $\sum_{n=1}^{N} \boldsymbol{\phi}_{n} \boldsymbol{\phi}_{n}^{\top}=\mathbf{\Phi}^{\top} \boldsymbol{\Phi}$ to obtain the normal equation for multiple outputs as follows:

<mark>$\mathbf{\Phi}^{\top} \mathbf{\Phi} \mathbf{W}^{*}=\mathbf{\Phi}^{\top} \mathbf{T}$</mark>

The same result can be obtained by expressing the loss function in a full vector notation without using the sum as we did in an earlier in Batch Learning: The Least Squares for Linear Regression Models. This time we need to be careful since we are dealing with multiple outputs that results in matrices $T$ and $Y(X,W)$ instead of a vectors $t$ and $y(X,w)$. To do so we will use matrices norms defined as $\|\mathbf{A}\|_{F}^{2}=\sum_{i=1}^{K} \sum_{j=1}^{K} a_{i, j}^{2}$. It is really nothing but squaring all the elements of a matrix and summing them up all together. *This is called Frobenius norm or Euclidian norm for matrices* (similar to Euclidian vector norm that we have been using so far). For simplicity of presentation we will just denoted as we denote a ususla vector norm, we can easily tell the difference due to using capital letters for matrices. Now the loss function can be expressed as:

(41) <mark>$\bar{J}(\mathbf{W})=\frac{1}{2 N}\|\mathbf{T}-\mathbf{Y}(\mathbf{X}, \mathbf{W})\|^{2}$</mark>

(42) <mark>$\bar{J}(\mathbf{W})=\frac{1}{2 N}\|\mathbf{T}-\mathbf{\Phi} \mathbf{W}\|^{2}$</mark>

We simply can obtain the gradient as:

<mark>$\nabla \bar{J}(\mathbf{W})=\frac{1}{2 N} \mathbf{\Phi}^{\top}(\mathbf{T}-\mathbf{\Phi} \mathbf{W})$</mark>

We set the gradient as usual to 0 and solve in order to obtain optimal solution $W^*$:

<mark>$\mathbf{\Phi}^{\top} \mathbf{\Phi} \mathbf{W}^{*}=\mathbf{\Phi}^{\top} \mathbf{T}$</mark>

Which is the same normal equation as above.

To see how the matrices are interacting with each other in terms of the dimensions we can add the dimension of each matrix so see how the intermediate dimensions are cancelled out during multiplication to end up with matrix of size ($M,K$) on both sides of the equation.

<mark>$\mathbf{\Phi}_{(\mathrm{M}, \mathrm{N})}^{\top} \mathbf{\Phi}_{(\mathrm{N}, \mathrm{M})} \mathbf{W}_{(\mathrm{M}, \mathrm{K})}^{*}=\mathbf{\Phi}_{(\mathrm{M}, \mathrm{N})}^{\top} \mathbf{T}_{(\mathrm{N}, \mathrm{K})}$</mark>

Ok, now we multiply by the inverse of $Φ^T Φ$ to get:

<mark>$\mathbf{\Phi}^{\top} \mathbf{\Phi} \mathbf{W}^{*}=\mathbf{\Phi}^{\top} \mathbf{T}$</mark>

(43) <mark>$\mathbf{W}^{*}=\left(\mathbf{\Phi}^{\top} \mathbf{\Phi}\right)^{-1} \mathbf{\Phi}^{\top} \mathbf{T}$</mark>

Again, the algorithm is similar to the least squares shown in Algorithm 1, but we are dealing with a matrix of weights $W$ instead of a vector of weights $w$. we will show you a one later once we develop the concept of regularisation for this general multi-output case.

##Sequential Learning: Multi-Output Stochastic Gradient Descent for Linear Regression Models with Basis

The stochastic gradient descent algorithm for multi-output regression can be written similar to one-output by adjusting the algorithm to deal with weights matrix instead of a weight vector. Essentially, we replace the loss function $\overline{\boldsymbol{J}}$ by $\boldsymbol{J}_{n}^{2}$, so after a data point becomes available, we update according to:

<mark>$\mathbf{W}^{(\tau+1)}=\mathbf{W}^{(\tau)}-\eta \frac{1}{2 N} \nabla J_{n}^{2}$</mark>

<mark>$\mathbf{W}^{(\tau+1)}=\mathbf{W}^{(\tau)}-\eta \frac{1}{N}\left(-\boldsymbol{\phi}_{n}\right)\left(\boldsymbol{t}_{n}-\mathbf{W}^{(\tau)^{\top}} \boldsymbol{\phi}_{n}\right)^{\top}$</mark>

(44) <mark>$\mathbf{W}^{(\tau+1)}=\mathbf{W}^{(\tau)}+\eta \frac{1}{N} \boldsymbol{\phi}_{n}\left(\boldsymbol{t}_{n}^{\top}-\boldsymbol{\phi}_{n}^{\top} \mathbf{W}^{(\tau)}\right)$</mark>

Where $τ$ represents the iteration (or the time step) and $η$ is the learning rate parameter which should be carefully chosen so that it does not lead to divergence or oscillation of the algorithms. The resultant algorithm is similar to Algorithm 2 but we are dealing with a matrix of weights $W$ instead of a vector of weights $w$.
