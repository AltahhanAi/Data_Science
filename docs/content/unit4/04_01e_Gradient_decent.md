#Approximate Solutions: Gradient Descent

For linear models we saw that we can find analytically a solution via the normal formula by setting the gradient to 0 and solve with respect to $w$, such solutions are either not available when we deal with non-linear optimisation or is not desirable due to efficiency requirements. Even if analytical close form solution is available, the complexity of finding the least squares is $\mathcal{O}\left(N^{3}\right)$ which is quite expensive when $N$ is reasonably large.

In such cases, it is desirable to find an approximate solution for the problem (i.e. an approximation for $w^*$) to come as close as possible to the minimum without necessarily finding the exact solution. Algorithms that tries to achieve this are called approximation algorithms, you will study several of these in the Algorithms Modules including greedy, local search and dynamic programming algorithm. In our case, we will utilise an important and pervasive approximation algorithm that is utilised throughout machine learning. It is not necessary the best approximation algorithm but it is the simplest to understand and to implement.

This optimisation algorithm is called the gradient descent or steepest descent. This techniques aims at iteratively finding the minimum of a function (the loss function $\bar{J}$ in our case). The algorithm starts from any point on the surface of the loss function (i.e. by taking a random initial value for $w$) and then it takes small steps in the direction of the minimum of the function by changing the weights gradually in each step. The direction of the point $w^*$ that minimise $\bar{J}$ from any point $\mathbf{w}^{(\tau)}$ is always opposite to the gradient of the function at this point $-\nabla \bar{J}\left(\mathbf{w}^{(\tau)}\right)$. This is because the gradient of a function always points in a direction opposite to the minimum.

Here we are talking about a minimum, often complex loss functions have several minima so we will come back to this idea later when we move to the non-linear models towards the end of the unit. We are taking small steps towards the minimum because taking large steps lead to overshooting the minimum or oscillating around it. The size of the step (denoted as $η$) is called the learning rate because it represents how fast a model can learn the solution of the problem. Gradient descent is a numerical optimisation technique so it is an iterative technique that keep working though iterations until it reaches a good enough approximate solution. Reaching a minimum is called convergence (a well know concept in calculus).
Let us start by the basic gradient descent update which takes the following form:

(6) <mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta_{\tau} \nabla \bar{J}(\mathbf{w})$</mark>

Where $\mathbf{w}^{(\tau)}$ represents the weight vector at iteration τ and this is not exponentiation. $η_τ$ is a learning step that can be varied between iterations to make the algorithms responsive to changes in the loss function terrain. The basic form of the GD is given below.

!!! example "**Algorithm 2:** Approximation Algorithm: Gradient Descent"
    **Input:**

    <mark>$\begin{array}{l}\text { Input set: } \mathbf{X}=\left\{\mathbf{x}_{1}, \ldots \mathbf{x}_{N}\right\} \\ \text { Labels } \underline{\underline{\text { set }}} \mathbf{t}=\left\{t_{1}, \ldots t_{N}\right\}\end{array} \mid$ *Training set*</mark>

    **Output:** $w$ an approximation for optimum weights $w^*$

    **GD**$(X,t)$:

    Initialise $w$

    For $\tau=1: \operatorname{tmax}$

    <mark>$\mathbf{w}=\mathbf{w}-\eta_{\tau} \nabla \bar{J}(\mathbf{w}, \mathbf{X}, \mathbf{t})$</mark>  # obtain the gradient of the loss for current $w$ on the entire training set.

    Return the final solution $w$.

There are some optimisation techniques that give us how to vary the learning rate $η_τ$ but we have not shown this her for simplicity. Also note that both $η_τ$ and $τmax$ should be inputs to the algorithm but we omit this to promote simplicity.

In the next section we will see how to apply the gradient descent algorithm on the linear regression model.

##Batch Gradient Descent for Linear Regression Models

We will take the gradient of the cost function directly without using it vectorised form but later we develop a vectorised version. We have saw already in a previous section that the gradient of the linear regression loss function takes the form:

<mark>$\bar{J}=\frac{1}{2 N} \sum_{n=1}^{N} J_{n}^{2}$</mark>

<mark>$\nabla \bar{J}(\mathbf{w})=\frac{1}{2 N} \sum_{n=1}^{N} \nabla J_{n}^{2}(\mathbf{w})$</mark>

<mark>$\nabla \bar{J}(\mathbf{w})=\frac{1}{2 N} \sum_{n=1}^{N} 2 \nabla J_{n}(\mathbf{w}) J_{n}(\mathbf{w})$</mark>

<mark>$J_{n}(\mathbf{w})=\left(t_{n}-\mathbf{w}^{\top} \mathbf{x}_{n}\right)$ and $\nabla J_{n}(\mathbf{w})=-\mathbf{x}_{n}$ hence</mark>

(5) <mark>$\nabla \bar{J}(\mathbf{w})=-\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_{n}\left(t_{n}-\mathbf{w}^{\top} \mathbf{x}_{n}\right)$</mark>

This is an important formula that we will get refer back to often. To get a taste of what this gradient entails, we show below what is involved in it:

<mark>$\nabla \bar{J}(\mathbf{w})=-\frac{1}{N}\left(\left[\begin{array}{c}x_{0} \\ x_{1} \\ \vdots \\ x_{D}\end{array}\right]_{1} J_{1}+\left[\begin{array}{c}x_{0} \\ x_{1} \\ \vdots \\ x_{D}\end{array}\right]_{2} J_{2}+\cdots+\left[\begin{array}{c}x_{0} \\ x_{1} \\ \vdots \\ x_{D}\end{array}\right]_{N} J_{N}\right)=0$</mark>

Therefore, the gradient descent algorithm for linear regression model, which acts on the entire training set, takes the form:

(7) <mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\frac{1}{N} \sum_{n=1}^{N} \eta \mathbf{x}_{n}\left(t_{n}-\mathbf{w}^{\top} \mathbf{x}_{n}\right)$</mark>

The number of iterations that we need to take in order to reach the minimum depends on $η$. The smaller $η$ is the more iterations we need to take, however we need to strike a balance here because if $η$ is too big then the algorithm might either oscillate or completely diverge (go away from the minimum). $η$ is almost always less than 1, a reasonable value of $η=0.01$ for linear regression. For other more complex models $η$ may need to take much smaller values. Each sweep through the entire dataset is called an epoch and this is a hyper parameter that we need to set, often between 10 and 100.

!!! example "**Algorithm 3:** Approximation Algorithm: Batch Gradient Descent (BGD) for Linear Regression Model, without vectorisation."
    **Input:**

    <mark>$\begin{array}{l}\text { Input set: } \mathbf{X}=\left\{\mathbf{x}_{1}, \ldots \mathbf{x}_{N}\right\} \text { each } \mathbf{x}_{n} \text { is a vector of size } D \\ \text { Labels } \underline{\underline{\text { set }}} \mathbf{t}=\left\{t_{1}, \ldots t_{N}\right\} \text { each } t_{n} \text { is a scalar }\end{array} \mid$ *Training set*</mark>

    $η$: The learning rate
    $epcs$: Max number of epochs

    **Output**: $w$ an approximation for optimum weights $w^*$; a vector of size $D+1$

    **BGD_LRegress** $(X,t,η,epcs)$:

    Initialise $w$ and set

    For $epoch = 1: epcs$

    <mark>$\mathbf{w}^{\prime}=\mathbf{0}_{D+1}$</mark>

    For $n=1:N$

    $\mathbf{x}_{n}=\left[1, \mathbf{x}_{n}^{\top}\right]^{\top}$   # add a dummy attribute for each $x_n$

    $\boldsymbol{w}^{\prime}=\boldsymbol{w}^{\prime}+\eta \boldsymbol{x}_{n}\left(t_{n}-\boldsymbol{w}^{\top} \boldsymbol{x}_{n}\right)$   # accumulated the changes without committing them

    $\mathbf{w}=\mathbf{w}+\frac{1}{N} \mathbf{w}^{\prime}$   # now commit the changes

    Return the final solution $w$.

Note how we accumulate the changes inside a temporary vector $w'$ (this is just a vector of size $D+1$) in an epoch and we commit at the end of the epoch. This is why it is called a batch gradient descent algorithm; we are waiting till the end of iterating through full batch of the dataset and then we change $w$, *i.e. we do not change $w$ during the epoch.*

This form of batch gradient descent does not take advantage of vectorisation and is slow. For large dataset vectorising the implementation is impractical. However, later on when we talk about mini-batch stochastic gradient descent we will see a way to make a good compromise that will allow us to utilise vectorisation.

For further reading, see Yoshua Bengio's paper on <a href="https://arxiv.org/pdf/1206.5533.pdf" target="_blank">Practical Recommendations for Gradient-Based Training of Deep Architectures</a>.

##Sequential Learning: Stochastic Gradient Descent for Linear Regression Models

Batch learning algorithm such as LS Regression or Batch Stochastic Gradient Descent take into account the entirety (the whole batch) of the dataset at once. No intermediate learning occurs. Another way to minimise the loss function is to gradually change the weights towards minimising the loss function instead of going all the way according to the sum of the errors. This is called sequential learning. There are several advantages for this approach. The most obvious advantage is that it allows for a stream of data to be fed into a system and the system can learn live as the data arrives from the stream. The main advantage is that learning can occur immediately for any fed sample and we do not need to wait to see the entirety of the dataset to learn a model.

Here we need to understand the concept of a learning rate or learning steps denoted as $η$. This hyper parameter specifies how much of the individual step error we want to take into account. In simple linear models this will not make a difference and in fact if assumed that the loss function is concave i.e. it has a global optimum then we can go all the way and adopt the entirety of each step error $\left(\mathbf{w}^{\top} \mathbf{x}_{n}-t_{n}\right)$ offline without changing the weights in each step. However, when the concavity of the loss function (existence of global optimum) is not guaranteed and when the loss function has several local optima some of which are really slight valleys (or when it is infested with local optima) then adopting the full error $t_{n}-\boldsymbol{y}\left(\mathbf{x}_{n}\right)$ is not a good idea. This is because it will force the model to fall into the nearest local minimum and consequent updates are spent on moving out or into local minima. Bearing in mind that the data is noisy anyway, we would want to utilise the learning step for our benefit to reduce the effect of the noise and help avoid the problem of overfitting. Essentially, we replace the loss function $J$ by $J_n^2$, so after a data point becomes available, we update according to:

<mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta \frac{1}{2} \nabla J_{n}^{2}$</mark>

<mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta\left(-\mathbf{x}_{n}\right)\left(t_{n}-\mathbf{w}^{(\tau)^{\top}} \mathbf{x}_{n}\right)$</mark>

(8) <mark>$\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta \mathbf{x}_{n}\left(t_{n}-\mathbf{w}^{(\tau)^{\top}} \mathbf{x}_{n}\right)$</mark>

Where $τ$ represents the iteration (or the time step) and η is the learning rate parameter which should be carefully chosen so that it does not lead to divergence or oscillation of the algorithms. This is because effectively we are accounting for only a very small part of the gradient and we need to leave room for gradients of other data points to take effect in later iterations. Note that we added a (-) to go against the gradient direction which will make the changes go in the direction that will minimise the error. A reasonable choice of $η$ is to make it proportional to the number of expected data points.

Stochastic Gradient Descent (SGD) suffers from several issues, mainly its high variance and its short-sighted look into the loss function terrain by considering just one or few data points. The gradient as a vector, has two essential properties that will direct our search for the minimum of the loss. The first is its direction and the second its magnitude. In general, variants of SGD optimisation use the gradient to specify the direction of changes in the weight vector, they vary in the way they estimate how much of the magnitude of the gradient to be considered (the amount of change). Other optimisation techniques use different direction than that of the gradient altogether, ex. the conjugate gradient of the loss, these however lie outside the scope of our coverage. Vanilla SGD just uses the learning rate $η$ to uniformly take a proportion of the gradient magnitude not all of it. But this makes it difficult to calibrate the learning rate because it has to fit all the different terrains of the loss function, so we normally end up reducing $η$ and taking lots of steps to converge to the minimum (of course there local and global minimum, but let us not differentiate for a moment).

There are a lot variations for the stochastic gradient descent. Mainly they are concerned with tailored learning rate that is responsive to the geometrical aspects of the loss function.

1. For example we can employ the idea of momentum which changes the learning rate according to a linear combination of the gradient at the current step with the previous update: called Momentum.

2. We can also employ the idea of averaging the weights from all past steps which do not play around with the learning rate, it replaces it by averaging of the weights themselves: called Averaging.

3. We can adapt the learning rate for each individual weight. Once way to do that is by dividing keeping a running average for each weight and then we divide the learning rate of each individual weight by the weight running average: called RMSProp.

4. we can also make the learning rate changes for each parameter according to how often the parameter is updated. This particularly applicable for when we have many zeros in our inputs for several attributes quite often- technical term is sparsity: this approach called Adagrad.

5. Another idea is to keep a running average of each individual weight but involve both the gradient and the square of the gradient (second moment) of the weight as well as the idea of sparsity in Adagrad: this approach is called adaptive moment-Adam optimisation and is very popular in deep learning. Standard optimisation libraries allow you to choose what type of optimisation you want to use without having to implement it from scratch.

For further reading, see Sebastian Ruder's paper on <a href="https://arxiv.org/pdf/1609.04747.pdf" target="_blank">An overview of gradient descent optimization algorithms</a>.

We will refer to all of these strategies by using a normalisation vector $\overline{\boldsymbol{N}}$ that can represent any of the above strategies. To cover the per-weight learning rate adaptation methods, such as the Adagrad and Adam, we need component-wise multiplication (denoted as *).  We write $\frac{1}{\bar{N}} * \mathbf{x}_{n}$ to express that we are adjusting the weights components differently, this is a crude way of describing these optimisations but promote simplicity.

For linear regression we can set $η$ to relatively high value such as 0.3 to take into account a good chunk of the errors since we know that the loss function is concave. The loss function is concave since we are taking the squares of weights with no activation function (we will talk more about activation function later in numerical classification). Still, we might want to use a reduced learning rate to cancel some of the noise of the data. Recall that any data will always have some noise in it and reducing the learning rate helps in reducing the risk of model overfitting and helps in reducing the effect of the noise. This is especially relevant when we talk about data streaming where we do not want to take into account all the error of the current input so as not undo completely some previous learning. Also, this brings us to the idea of input normalisation which should be used if possible, for input coming from data streams.

The idea of a learning step is pervasive in machine learning and can be powerful in tackling some of the overfitting issues that arise when dealing with regression. For example, we can anneal (gradually reduce) the learning step in each step or every b steps in order to hinge towards a global optimum when the loss function is not concave.

Another important reason to use SG Regression is that it is often faster to converge in practice than LS when we deal with more complex techniques and is more efficient to implement when the size of the dataset or its dimensionality is intractable. This is only for extremely large dataset but something worth putting in mind for future reference.

We can also apply SGD regression on a static dataset, we get a similar result to the LS regression. However, you should be bear in mind that there are quite subtle differences between the two algorithms (the LS Regression and SG Regression). Let us see first SGD Regression on a dataset below.

!!! example "**Algorithm 4:** Sequential Learning: Stochastic Gradient Descent for Linear Regression Model."
    **Input:**

    <mark>$\begin{array}{l}\text { Input set: } \mathbf{X}=\left\{\mathbf{x}_{1}, \ldots \mathbf{x}_{N}\right\} \operatorname{each} \mathbf{x}_{n} \text { is a vector of size } D \\ \text { Labels } \underline{\underline{\text { set }}} \mathbf{t}=\left\{t_{1}, \ldots t_{N}\right\} \text { each } t_{n} \text { is a scalar }\end{array} \mid$ *Training set*</mark>

    $η$: The learning rate

    $epcs$: Max number of epochs

    **Output**: $w$ an approximation for optimum weights $w^*$; a vector of size $D+1$

    **SGD_LRegress**$(X,t,η,epcs)$:

    Initialise $w$

    For epoch = 1: *epcs*

    $For n=1:N$

    $\mathbf{x}_{n}=\left[1, \mathbf{x}_{n}^{\top}\right]^{\top}$   # add a dummy attribute for each $x_n$

    $\mathbf{w}=\mathbf{w}+\eta \boldsymbol{x}_{n}\left(t_{n}-\boldsymbol{w}^{\top} \boldsymbol{x}_{n}\right)$   # commit the changes in every step

    Return the final solution $w$

Comparing Algorithm 3 and Algorithm 4. It becomes clear that in Algorithm 4 the weights fixed *during* learning. In contrast Algorithm 3 accumulates all the changes of the weights and apply them all at once.

<figure role="group">
  <img src="../images/DS_IMG011.png" alt="Test image." />
  <figcaption><strong>Figure 4.6.</strong> SGD algorithm behaviour: SGD takes gradual steps towards the minimum of the loss function by  following the gradient of the loss. The line shows an example of the paths of a batch gradient descent (blue on the loss surface function and its projection is orange on the loss contour) and stochastic gradient descent algorithms(green on the loss surface function and brown on the loss contours). </figcaption>
</figure>

##Minim-Batch Learning: Mini-Batch Stochastic Gradient Descent for Linear Regression Models

Mini-batch SGD algorithm can be used to reach something in the middle between sequential and batch gradient methods. To achieve this: the weights changes can be accumulated on-the-sides not for the entire training set but for a limited number of steps b and be committed every b steps. On the extremes when we set $b=N$ (wait until all the data finishes) we get an algorithm that is equivalent to the batch gradient descent, while when we set $b=1$ we get the stochastic gradient descent. So $b$ parametrise a middle ground approach for both cases. Below we show the algorithm.

<mark>Watch a video1 that explains the above concepts</mark>

!!! example "**Algorithm 5:** Mini-Batch Stochastic Gradient Descent Updates for Linear Regression Model: Vanilla implementation that can be sped up- see next algorithm."
    **Input:**

    <mark>$\begin{array}{l}\text { Input set: } \mathbf{X}=\left\{\mathbf{x}_{1}, \ldots \mathbf{x}_{N}\right\} \text { each } \mathbf{x}_{n} \text { is a vector of size } D \\ \text { Labels } \underline{\underline{\text { set }}} \mathbf{t}=\left\{t_{1}, \ldots t_{N}\right\} \text { each } t_{n} \text { is a scalar }\end{array} \mid$ *Training set*</mark>

    $η$: The learning rate

    $b$:The mini-batch size (specifies how frequently we want to update the weights $w$).

    $epcs$: Max number of epochs

    **Output**: $w$ an approximation for optimum weights $w^*$; a vector of size $D+1$

    **SGD_MiniB_LRegress**$(X,t,b,η)$:

    Initialise $w$ and set $\mathbf{w}^{\prime}=0$

    For $epoch = 1: epcs$

    For $n=1:N$

    $\mathbf{x}_{\boldsymbol{n}}=\left[\mathbf{1}, \mathbf{x}_{\boldsymbol{n}}^{\top}\right]^{\top}$    # add a dummy feature for each $x_n$

    $\mathbf{w}^{\prime}=\mathbf{w}^{\prime}+\eta \mathbf{x}_{n}\left(t_{n}-\mathbf{w}^{\top} \mathbf{x}_{n}\right)$   # per-weight optimisation methods use $\frac{1}{\bar{N}} * \mathbf{X}_{n}$

    If $n \% b==0$   # there is a better condition see the discussion below

    $\mathbf{w}=\mathbf{w}+\frac{1}{b} \mathbf{w}^{\prime}$

    $\mathbf{w}^{\prime}=\mathbf{0}$

    Return the final solution $w$

The above algorithm utilises the modulus function $n%b$ which will give us the remainder of a division of the data point count $n$ by $b$ the batch size. When this function $n%b== 0$ it means that $b$ number of steps has elapsed. For example if $N=90$ and we set b=10 then the weights $w$ (not $w'$) will be updated every 10 steps and the updates will be executed 9 times. Note that we accumulate all the changes inside each 10 steps in $w'$ and we commit them at the 10th step.

If we have $N=94$ and we set $b=10$ then the last 4 data points will be left if we just use the condition $n%b==0$. Therefore, we can adjust as follows:

If $n \% b==0$ or $n==N:$

If $n \% b \neq 0$ then $b^{\prime}=n \% b$   # accommodate the last few points that do not fit a batch

Else $b^{\prime}=b$

$\mathbf{w}=\mathbf{w}+\frac{1}{b} \mathbf{w}^{\prime}$

$\mathbf{w}^{\prime}=\mathbf{0}$

The condition $n%b==0$  or  $n==N$ is used to accommodate the last few points that cannot form a full batch. For the same reason we use $b^{\prime}=n \% b$ instead of $b$, which will yield $b$ when $n \% b==0$ and the remainder of the batch (4 in our example) otherwise when $n==N$. we have not include this snippet to keep the algorithm simple.

Also, we use the weights $w$ in our output estimation $\mathbf{w}^{\top} \mathbf{x}_{n}$ in the update rule $\mathbf{w}^{\prime}=\mathbf{w}^{\prime}+\frac{1}{N} \eta\left(t_{n}-\mathbf{w}^{\top} \mathbf{x}_{n}\right) \mathbf{x}_{n}$ this is deliberate to guarantee stability. Mini-batch stochastic gradient descent reduce the variance caused by stochastic gradient descent and hence help stabilise the learning process.

###Faster Mini-Batch SGD

The above algorithm is a vanilla algorithm of an SGD mini-batch that can be sped up. The algorithm slows down the process by processing the data points individually in each batch. There is a better approach, can you guess it, take a minute or two to think about it…Ok, it is vectorisation. The idea is as follows. We assume that the size of the batch is $b$ and for simplicity that $N$ is divisible by $b$ the number of mini-batches as $q=N / b$ then. Now we can adopt one of two strategies:

1 - Define a partition of the training set ${X,t}$ into a set of q mini-batches as follows:

<mark>$X=[█(⏞([■(x_1,1&x_1,2&…&x_(1,D)@x_2,1&x_2,2&…&x_(2,D)@⋮&⋮&⋮&⋮@x_(b,1  )&x_(b,2  )&…&x_(b,D  ) )] )┴(X_1 )@ @ @⏞([■(…&…&…&…@…&…&…&…@⋮&⋮&⋮&⋮@x_(2b,1)&x_(2b,2)&…&x_(2b,D) )] )┴(X_2 )@ @⋮@ @⏞([■(…&…&…&…@…&…&…&…@⋮&⋮&⋮&⋮@x_(qb,1)&x_(qb,2)&…&x_(qb,D) )] )┴(X_q ) )]=[█(X_1@X_2@⋮@X_q )]=[X_1^⊤,X_2^⊤,…,X_q^⊤ ]^⊤ ,	t=[█(⏞([█(t_(1 )@t_2  @⋮@t_(b )  )] )┴(t_1 )@ @⏞([█(…@…@…@t_2b )] )┴(t_2 )@ @⋮@⏞([█(…@…@…@t_qb )] )┴(t_q ) )]=[█(t_1@t_2@⋮@t_q )]"=" [t_1^⊤,t_2^⊤,…,t_q^⊤ ]^⊤$<mark>

Where $X_τ τ=1:q$ is a matrix of size $b×D$ and $t_τ τ=1:q$ is a vector of size $b×1$. We refer to both as a mini-batch of size $b$.

So, now we sweep through all the mini-batches one after the other in each iteration to cover the whole training set, we call this an epoch. After each epoch we need to shuffle the dataset (or equivalently shuffle the membership assignment in the mini-batches which is what we always do in the implementation). Note that our weights estimation are expected to improve from one batch to another. Moreover, the weights error (cost function) is expected to improve from one epoch to another since we employ normally a learning rate<1. This strategy guarantees stability and efficiency at the same time. We will refer to this strategy as shuffling and partitioning strategy. Note that in this strategy each data point must appear once in one of the min-batches in each epoch.

2 - The second strategy is just to draw a random mini-batch $X_τ$ and $t_τ$ of size b without partitioning which is even more efficient than the first strategy. This is drawing with replacement so the same data point can appear in multiple mini-batches inside the same epoch or may not appear at all in any mini-batch (but the same data point does not appear more than once in the same mini-batch). In this strategy we can decouple the number of mini-batches from the size of the training set, so $q \geq N / b$ but it can still provide a general guide on the number of iterations (or mini-batches) the algorithm will go through. This strategy is faster in implementation, but it may lead to less stability than the first strategy, so it is more preferred in large scale learning. It all depends on selecting the trade-off that suits the application.

Both strategies are amenable for parallelisation, where we feed parts of the process to a different processor. Clearly, we can feed parts of a batch to different core processor but we cannot give different batches to different processor because the idea of the mini-batch is to update the weights directly after each batch and then use the new weights in the next batch. If we give away on this idea then we can distribute different batches on different processors and collect and aggregate the changes afterwards. Such a process will take us back to batch gradient descent. So, the implementation is similar to a min-batch but the effect is a batch gradient descent.

Both strategies are equivalent in the extremes: when b=1 or when $b=N$. when $b=1$ the SGD mini-batch algorithm turn into an SGD algorithm. When $b=N$ the SGD mini-batch algorithm becomes a batch SGD algorithm which in turn approximates the least squares but without the overhead of matrix inversion.

Below we show the final mini-batch algorithm. We have left which strategy to adopt open in the algorithm and we have stated it as ‘Select a mini-batch $X_τ,t_τ$ of size $b$ from $\mathbf{X}, \mathbf{t}: q \geq N / b$ (randomly or by shuffling and partitioning)’. So, this algorithm is actually two different algorithms depending on whether we choose to shuffle and partition the training set or just simply draw a random mini-batch (with replacement).

We normally decay the learning rate in order to prevent zigzagging around the minimum of the cost function when we start by a high learning rate or to fine tune our final weights.

!!! example "**Algorithm 4:** Mini-Batch Stochastic Gradient Descent Updates for Linear Regression Model: with vectorisation."
    **Input:**

    <mark>$\begin{array}{l}\text { Input set: design matrix } \mathbf{X}=\left[\mathbf{x}_{1}^{\top}, \ldots, \mathbf{x}_{N}^{\top}\right]^{\top} \text { each } \mathbf{x}_{n} \text { is a vector of size } D \\ \text { Labels } \underline{\underline{\text { set: }}} \mathbf{t}=\left[t_{1}, \ldots, t_{N}\right]^{\top} \text { each } t_{n} \text { is a scalar } \end{array} \mid$ *Data set*</mark>

    $b$: The mini-batch size (specifies how frequently we want to update the weights $w$).

    $η_0$: Initial learning rate

    $epcs$: Number of epochs

    **Output**: $w$ an approximation for optimum weights $w^*$; a vector of size $D+1$

    **SGD_VMiniB_LRegress**$(X,t,b,η_0,epcs)$:

    Initialise $w$ and $η=η_0$

    For $epoch = 1:epcs$

    For iteration $τ=1:q$    # $q≥N/ b$

    Select a mini-batch $X_τ,t_τ$ of size $b$ from $X,t$  	# by sampling or by shuffling & partitioning

    $\mathbf{X}_{\tau}=\left[\mathbf{1}_{b}, \mathbf{X}_{\tau}\right]$   # add dummy feature to the mini-batch

    $\mathbf{w}=\mathbf{w}+\eta \frac{1}{b} \mathbf{X}_{\tau}^{\top}\left(\mathbf{t}_{\tau}-\mathbf{X}_{\tau} \mathbf{w}\right)$   # per-weight optimisation methods use $\frac{1}{\bar{N}} * \mathbf{X}_{\tau}$

    Decay $η$   # if necessary: ex. ‎$η=0.9×η$

    Return the final solution $w$

Stochastic gradient descent, especially the minim-batch version is an excellent tool to tackle large-scale learning and has received recently a considerable attention. Large scale learning is learning from a large dataset with a huge amount of instances available. In such a case, we cannot expect to train on the whole dataset because it way beyond a single machine capabilities or even the capabilities a medium cluster of machines. Stochastic gradient descent is an excellent tool because it allows us to simply tackle as much as we can digest in our available hardware. Of course there are many augmentation to the simple (vanilla) SGD in terms of cleverer selection of the data to be processed which goes beyond the basic form presented here. At the same time, parallelisation techniques can be employed to promote efficient parallelised implementation of amortised complexity of $\mathcal{O}\left(\log _{r} N\right)$ where $N$ is the dataset size or the size of the processed data that can be pulled from a data lake or a data centre and $r$ is the number of parallel processors available to the algorithm.

It should be stressed here also that SGD is sensitive to feature scaling and it is recommend to scale or normalise all features of the our data to avoid one feature overwhelming other features by its high values that do not vary that much as we have discussed in unit1. The simplest way is to scale by dividing over the max value of the feature, or to subtract the minimum and divide by the difference between the minimum and the maximum of the feature. We get the min and max either by looking into the dataset or by understanding the domain of the feature what are those would. Consulting domain experts who works with the data is also useful to understand the nature of the dataset that we are tackling.

For further reading, see Prateek et al's paper on <a href="https://www.jmlr.org/papers/volume18/16-595/16-595.pdf" target="_blank">Parallelizing Stochastic Gradient Descent for Least Squares Regression: Mini-batching, Averaging, and Model Misspecification</a>.

The above algorithm can be easily adapted when we are dealing with a data stream, all what we need to do is to accumulate $X_τ$ as the data arrives until it is of the required size $b$, and we can even vary the size $b$ itself between different iterations, these have not been shown to keep the algorithm simple and to concentrate on a basics of the vectorised minim-batch is left to you as an exercise. Not that when $b=N$ the algorithm goes back to a vectorised batch stochastic gradient descent which can be applied when the dataset size permits. The resultant weights are still an approximation even if it might be very close to the optimum solution w^*.

To summarise, we emphasise here, contrary to what one might expect, stochastic and mini-batch stochastic gradient descent converge faster that batch gradient descent in practice. This is due to several reasons. One reason is that both stochastic gradient algorithms infuse noise in the update which is quite useful to escape local minima. Another reason is that by nature stochastic algorithms are faster to execute and they execute several updates per clock time in comparison with batch gradient which keeps accumulating the gradients on the side until it sweeps through the whole training set. Assuming that the training set is finite but large, then the roughness of stochastic updates outperforms the more exactness of batch gradient. A third reason is that all gradient descents, even the batch one, do not point exactly to the global minimum instead they roughly point to a direction that will lead us to the minimum. Therefore, it does not make sense to spend a lot of computational power (as in the batch GD) to try to improve the gradient by considering more and more points until we consume the whole training set. Because even then the gradient is not quite right opposite to the direction of the minimum for complex loss function. Although linear regression loss function is quadratic and has a global minimum, nevertheless these issues can still be seen and you can examine them in the next exercise.
