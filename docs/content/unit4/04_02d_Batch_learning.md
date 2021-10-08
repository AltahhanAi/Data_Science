#Batch Learning

##The Least Squares for Linear Regression Models with Fixed Basis

In this section we will derive how to train a linear regression model that uses some basis. In general, we will use the same formulation that we used earlier for training a regression model without basis. So, we have the same loss function as before but the model is expressed in terms of basis instead of the input features. We would need to adjust our estimations $y(\mathbf{x})$ by adjusting the parameters $\mathbf{w}$ to give us the desired answers $t$. Our model is written as:  

$$
y(\mathbf{x}, \mathbf{w})=\mathbf{w}^{\top} \boldsymbol{\phi}(\mathbf{x})
$$

Where $\mathbf{w}=\left(w_{0}, w_{1}, w_{2}, \ldots, w_{M}\right)$ and $\boldsymbol{\phi}=\left(\phi_{0}, \phi_{1}, \phi_{2}, \ldots, \phi_{M}\right)^{\top}$ and $\phi_{0}=1$.

For a data point $\mathbf{x}_{n}$ we have:

$$
y(\mathbf{x}, \mathbf{w})=\mathbf{w}^{\top} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)
$$

For brevity we will refer to $\boldsymbol{\phi}\left(\mathbf{x}_{n}\right)$ as $\boldsymbol{\phi}_{n}$ so the above is expressed as:

$$
y(\mathbf{x}, \mathbf{w})=\mathbf{w}^{\top} \boldsymbol{\phi}_{n}
$$

And the loss function can still be written as:

$$
\overline{J^{2}}(\mathbf{w})=\frac{1}{2 N} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, \mathbf{w}\right)\right)^{2}
$$

$$
\overline{J^{2}}(\mathbf{w})=\frac{1}{2 N} \sum_{n=1}^{N}\left(t_{n}-\mathbf{w}^{\top} \boldsymbol{\phi}_{n}\right)^{2}
$$

We define the **design matrix** $\Phi$, which has a dimension of $N×M$ and whose $\mathrm{n}^{\text {th }}$ row is $\boldsymbol{\phi}_{n}^{\top}$ and $\mathbf{t}$ is the target matrix, as follows:

$\boldsymbol{\Phi}=\left[\begin{array}{c}\boldsymbol{\phi}_{1}^{\top} \\ \vdots \\ \boldsymbol{\phi}_{n}^{\top} \\ \vdots \\ \boldsymbol{\phi}_{N}^{\top}\end{array}\right]=\left[\begin{array}{c}\boldsymbol{\phi}\left(\mathbf{x}_{1}\right)^{\top} \\ \vdots \\ \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\top} \\ \vdots \\ \boldsymbol{\phi}\left(\mathbf{x}_{N}\right)^{\top}\end{array}\right]=\left[\begin{array}{cccc}1 & \phi_{1}\left(\mathbf{x}_{1}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{1}\right) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & \phi_{1}\left(\mathbf{x}_{n}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{n}\right) \\ \vdots & \vdots & \ddots & \vdots \\ 1 & \phi_{1}\left(\mathbf{x}_{N}\right) & \cdots & \phi_{M}\left(\mathbf{x}_{N}\right)\end{array}\right] \quad \mathbf{t}=\left[\begin{array}{c}t_{1} \\ \vdots \\ t_{n} \\ \vdots \\ t_{N}\end{array}\right]$ and of course $\mathbf{w}=\left[\begin{array}{c}w_{1} \\ w_{2} \\ \vdots \\ w_{D}\end{array}\right]$

Similar to what we did in the previous section for the simple linear models we can arrive to the normal equations for the feature space by replacing $\mathbf{X}$ with $\boldsymbol{\Phi}$. All predictions of the model can be expressed as:

$$
\mathbf{y}(\mathbf{X}, \mathbf{t})=\mathbf{\Phi} \mathbf{w}
$$

The loss function can be written as a vector norm $\|.\|^{2}$ as follows:

$$
\overline{J^{2}}=\frac{1}{2 N}\|\mathbf{t}-\mathbf{\Phi} \mathbf{w}\|^{2}
$$

By taking the gradient and setting it to 0 we get:

$$
\nabla \overline{J^{2}}=\frac{2}{2 N} \mathbf{\Phi}^{\top}(\mathbf{t}-\mathbf{\Phi} \mathbf{w})=0
$$

$$
\mathbf{\Phi}^{\top} \mathbf{\Phi} \mathbf{w}^{*}=\mathbf{\Phi}^{\top} \mathbf{t}
$$

We multiply both sides by the inverse of $\mathbf{\Phi}^{\top} \mathbf{\Phi}$ to get:

$$
\mathbf{w}^{*}=\left(\mathbf{\Phi}^{\top} \mathbf{\Phi}\right)^{-1} \mathbf{\Phi}^{\top} \mathbf{t}
$$

The above gives us a closed form solution for $\mathbf{w}^{*}$ and is known as the normal equations. Closed form solutions are not always available for a machine learning or data mining tasks. Their existence facilitates more analysis and insights into the problem. Some problems might not have a closed form formula of their solution however we can still estimate the solutions numerically. Sometimes also closed form solution can be impractical for big datasets due to their computational demanding nature.

The algorithm that returns the optimal solution for a linear model is given as follows:

!!! info "Algorithm 1': Least Squares for Linear Regression Model ‎with Gaussian Basis"

    **Input:**

    Input set: design matrix $\mathbf{X}=\left[\mathbf{x}_{1}^{\top}, \ldots, \mathbf{x}_{N}^{\top}\right]^{\top}$ each $\mathbf{x}_{n}$ is of size $D$

    Labels set: vector $\mathbf{t}=\left[t_{1}, \ldots, t_{N}\right]^{\top}$ each $t_n$ is a scalar

    $\boldsymbol{\mu}_{j}$: M Basis centres, each is a vector of size $D$

    $\boldsymbol{\Sigma}$: Covariance matrix of size $D×D$

    **Output**: $\mathbf{w}^{*}$ optimum weights; a vector of size $M+1$

    **LS_LRegressBasis**$(\mathbf{X}, \mathbf{t}, \boldsymbol{\mu}, \mathbf{\Sigma})$:

    !!! quote ""

        Map the data $\mathbf{X}$ into design matrix $\boldsymbol{\Phi}$ via the Gaussian basis $\phi_{j}\left(\mathbf{x}_{n}\right)=e^{-\frac{1}{2}\left(\mathbf{x}_{n}-\mu_{j}\right)^{\top} \Sigma^{-1}\left(\mathbf{x}_{n}-\mu_{j}\right)}$

        !!! quote ""
            $\mathbf{\Phi}=\left[\mathbf{1}_{N}, \mathbf{\Phi}\right]$ <span style="float: right;"># add dummy feature to the design matrix</span>

            $\mathbf{w}^{*}=\left(\mathbf{\Phi}^{\top} \mathbf{\Phi}\right)^{-1}\left(\mathbf{\Phi}^{\top} \mathbf{t}\right)$



        **Return** solution $\mathbf{w}^{*}$

Note that the basis is fixed and not changed during learning. This a key difference between linear and non-linear models which has adaptive basis.

###Complexity discussion

The same discussion that we had earlier applies again of course for the case of feature space but this time the complexity is in terms of $M$. The Least Squares on linear regression with basis has the same complexity as in the simple regression without basis. The only difference is that it will be related to $M$ the feature space dimension instead of $D$ the input space dimension.

##Batch Learning, Sequential or Mini-Batch Stochastic Learning:

The least square solution is called batch solution since they dictate processing of the entire dataset at once (see Algorithm1) specifically in terms of the Design matrix $Φ$. This restricts the applicability of the algorithms and makes it difficult and computational costly to apply them on a big dataset. This is when the stochastic (or sequential) gradient decent comes to the rescue. Essential all the three algorithms that we have considered for linear models on input X also apply for the feature space. Therefore, for brevity we will only show the vectorised stochastic gradient descent. We consider one (or more mini-batch) data point at a time and we update the weights accordingly without waiting until all the other updates becomes available. This is particularly useful when the data is fed from a stream or when we are tackling large-scale learning.  

!!! info "Algorithm 6': Mini-Batch Stochastic Gradient Descent Updates for Linear Regression Model: with vectorisation and basis functions."

    **Input:**

    Input set: design matrix $\mathbf{X}=\left[\mathbf{x}_{1}^{\top}, \ldots, \mathbf{x}_{N}^{\top}\right]^{\top}$ each $\mathbf{x}_{n}$ is a vector of size $D$ <span style="float: right;">*Training set*</span>

    Labels set: vector $\mathbf{t}=\left[t_{1}, \ldots, t_{N}\right]^{\top}$ each $t_{n}$ is a scalar <span style="float: right;">*Training set*</span>

    $b$:The mini-batch size (specifies how frequently we want to update the weights $\mathbf{w}$)

    $\eta_{0}$: Initial learning rate

    $epcs$: Max number of epochs

    $\boldsymbol{\mu}_{j}$: $M$ Basis centres, each is a vector of size $D$

    $\boldsymbol{\Sigma}$: Covariance matrix of size $D×D$

    **Output**: $\mathbf{w}$ an approximation for optimum weights $\mathbf{w}^{*}$; a vector of size $M+1$

    **SGD_LRegressBasis**$\left(\mathbf{X}, \mathbf{t}, b, \eta_{0}, e p c s\right)$:

    !!! quote ""
        Initialise $\mathbf{w}$ and $\eta=\eta_{0}$

        For epoch = $1:epcs$

        !!! quote ""
            For iteration $τ=1:q$ <span style="float: right;"># $q≥N/ b$</span>

            !!! quote ""

                Select a mini-batch $\mathbf{X}_{\tau}, \mathbf{t}_{\tau}$ of size $b$ from $\mathbf{X}, \mathbf{t}$ <span style="float: right;"># *By sampling or by shuffling & partitioning*</span>

                Map $\mathbf{X}_{\tau}$ to $\mathbf{\Phi}_{\tau}$ <span style="float: right;"># $\phi_{j}\left(\mathbf{x}_{n}\right)=e^{-\frac{1}{2}\left(\mathbf{x}_{n}-\mu_{j}\right)^{\top} \Sigma\left(\mathbf{x}_{n}-\mu_{j}\right)} j=1, \ldots, M$</span>

                $\mathbf{\Phi}_{\tau}=\left[\mathbf{1}_{b}, \mathbf{\Phi}_{\tau}\right]$ <span style="float: right;"># add dummy feature to the mini-batch</span>

                $\mathbf{w}=\mathbf{w}+\eta \frac{1}{b} \mathbf{\Phi}_{\tau}^{\top}\left(\mathbf{t}_{\tau}-\mathbf{\Phi}_{\tau} \mathbf{w}\right)$

            Decay $\eta$ <span style="float: right;"># if necessary: ex. $‎η=0.9×η$</span>

        Return the final solution $\mathbf{w}$
