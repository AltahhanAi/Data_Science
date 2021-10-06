#Linear Regression with Linear and Non-Linear Basis Functions

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * generalise linear regression models with basis
    *	understand different types of basis functions that can be utilised in the generalised linear regression models
    *	understand the limits of generalised linear models in terms of their ability to model non-linear functions
    *	perform batch and sequential learning in generalised linear models
    *	understand overfitting and underfitting in generalised regression models
    *	understand generalised multi output linear regression models.

In the previous lesson we saw how to minimise a loss function of a linear model of the form:

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{D} x_{D}
$$

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\sum_{d=1}^{D} w_{d} x_{d}
$$

But what about if we wanted to process the data $\mathbf{x}$ before we try to learn a model? This is called input space mapping, i.e. we would like to map the input space $\mathbf{X}$ to some other space $𝚽$.  We do that when we perform some pre-processing on the input space but also when we want to increase or decrease the dimensionality of the input space. There are numerous advantages of moving from one space to the other in data mining; it all amounts to simplifying or reducing the complexity of the data or its processing.

Learning takes place by adjusting these parameters to make the model produce the desired answers. In simple terms, linear means first order sum. So, the above is linear because the model constitutes a linear function in the weights. For example, the following **is not a linear model**: $y(\mathbf{x}, \mathbf{w})=w_{0}^{2}+w_{1}^{2} x_{1}+w_{2}^{2} x_{2}+\cdots+w_{D}^{2} x_{D}$, because we are multiplying each component of $x_i$ by the square of the weight $\left(w_{n}\right)^{2}$. Similarly, $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1}^{3} x_{1}+w_{2} x_{2}$ is not a linear model. Polynomials of order higher than 1, quadratic, cubic as well as exponential $e^x$, logarithmic, sin, cos are all nonlinear functions).

We still call $w_0$ the bias or the intercept. Note that given an input space with one input $x_1$ our linear model can be defined as $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}$.

However, the following are all **linear models**:

- $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}^{2}+w_{2} x_{2}^{2}+\cdots+w_{D} x_{D}^{2}$

- $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}^{1}+w_{2} x_{2}^{2}+\cdots+w_{D} x_{D}^{D}$

- $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} e^{x_{1}}+w_{2} e^{x_{2}}+\cdots+w_{D} e^{x_{D}}$

- $y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} e^{x_{1}}+w_{2} e^{x_{2}}+w_{3} e^{x_{1}+x_{2}}+w_{3} e^{x_{1}-x_{2}}$

!!! note
		Remember our note about being linear with respect to the weights. Since this expression is a linear combination of the weights, the linearity of the model holds).

		Note that in the last example we have 5 weights although the input space has a 2 features $x_1$ and $x_2$.

More generally, we can define a linear model using a set of $M$ functions called basis. Each maps an input vector $\mathbf{x}=\left(x_{1}, x_{2}, \ldots, x_{D}\right)^{\top}$ into a real value $\phi_{m}(\mathbf{x})$ i.e. $\phi_{m}: \mathbb{R}^{\mathrm{D}} \longrightarrow \mathbb{R}$. Then, we can use each set of basis functions to map the input space into a new input space, and our linear regression model can be expressed as:

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} \phi_{1}(\mathbf{x})+w_{2} \phi_{2}(\mathbf{x})+\cdots+w_{M} \phi_{M}(\mathbf{x})
$$

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\sum_{m=1}^{M} w_{m} \phi_{m}(\mathbf{x})
$$

Note that the sum has $M$ elements according to the number of basis functions that we use and we have also a set of $M$ weights (including $w_0$) instead of the $D$ weights that we had earlier when we were dealing directly with the components of the input space. Note that the linear models without basis become a special case of linear models with basis $\phi_{m}(\mathbf{x})=x_{m}$ here $M=D$. So, we will be dealing with the more general definition of the linear model from now on. Note that in effect, the set of basis functions defines a set of components of a vector $\boldsymbol{\phi}(\mathbf{x})=\left(\phi_{1}(\mathbf{x}), \phi_{2}(\mathbf{x}), \ldots, \phi_{M}(\mathbf{x})\right)^{\top}$, which in turn means that $\phi: \mathbb{R}^{\mathrm{D}} \longrightarrow \mathbb{R}^{\mathrm{M}}$. For brevity, we write $\phi=\left(\phi_{1}, \phi_{2}, \ldots, \phi_{M}\right)^{\top}$ when we are not concerned in explicitly stating $x$.

Moving from input space to feature space has several desired advantages. The main one is that while moving to a high dimensionality may entail some extra processing, it can simplify the model that is needed in order to fit the data. Specifically, in moving to higher dimensionality feature space we hope to map a non-linear relationship between the input and the label into a linear relationship between the features and the label. This trick allows us to employ simpler models and promote speed and efficiency. Even when the relationship does not become linear, moving to a feature space can make it simpler.

##Vector Matrix Representation with Dummy Component

Similar to what we did earlier for the input space we can also vectorise the features space. Looking at the above linear model expressions, we can easily identify that they can be changed into:

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} \phi_{1}(\mathbf{x})+w_{2} \phi_{2}(\mathbf{x})+\cdots+w_{M} \phi_{M}(\mathbf{x})
$$

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\mathbf{w}^{\top} \boldsymbol{\phi}
$$

However, it is also more useful if we express the whole model using vectors. To do so, we can define a **dummy** feature $\phi_{0}=1$ for all vectors of the input space. In this case we can define our linear model as:

$$
y(\mathbf{x}, \mathbf{w})=\mathbf{w}^{\top} \boldsymbol{\phi}
$$

Where $\mathbf{w}=\left(w_{0}, w_{1}, w_{2}, \ldots, w_{M}\right)$ and $\boldsymbol{\phi}=\left(\phi_{0}, \phi_{1}, \phi_{2}, \ldots, \phi_{M}\right)^{\mathrm{T}}$ and $\phi_{0}=1$.

<figure role="group">
  <img src="../images/DS_IMG104.jpg" alt="Schematic representation of a linear regression model with basis." />
  <figcaption><strong>Figure 4.7.</strong> Schematic representation of a linear regression model with basis.</figcaption>
</figure>

By fixed basis we mean that the basis function does not change. However, this does not mean that it is constant, it means the basis function itself does not change from one function to another-in terms of type and parameters. So we can use a fixed basis to map the input space into a new features space and then use the features to learn a linear model. The resultant model is still linear in the feature space.
