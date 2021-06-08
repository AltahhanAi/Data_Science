# Linear and non-linear regression

!!! success "Learning outcomes:"
	After completing this unit you should be able to:

    * use linear regression models in order to predict the value of a continuous label (regression)
    * understand how we can establish objective functions to optimise a linear model
    *	devise an exact solution for linear regression models through least squares
    *	devise an inexact solution for linear regression models through gradient descent
    *	understand how generalised linear models utilise basis functions in order to obtain the capabilities of modelling a non-linear function
    *	further develop the generalised linear regression models into multi output generalised linear regression models
    *	devise a multi-layer neural network learning algorithm with one output, that is capable of automatically inferring the best basis for a regression problem.

**In Unit 2, you saw how we can apply decision trees to classify data with tree induction algorithms such as CART. When the data is purely numerical (all features are numerical not categorical or nominal) then DT might not be the best technique to choose. Similarly, although we have not showed it yet, DT can be used for regression, but by nature DT in its simplest form needs to utilise some from of discretisation for regression and will normally perform inferior to other techniques that are originated form numerical optimisation.**

In this unit, we will cover models that allow us to perform regression and classification on numerical datasets.  Even if the dataset is not fully numerical, we can convert those features that are categorical into numerical values without loss of accuracy or generality. Of course, this will depend on the feature's values and its nature. A common feature is gender where we can convert its values into {0, 1, 2} to represent male, female, and no specific gender, respectively. Those models also output numerical values. For regression, this would be exactly what we want since regression is concerned with predicting a specific value instead of a category. For classification, we can either take the output to mean probability of belonging to a specific class (and in this case we would need multi-output that represents each possible class) or we can convert it via other discretisation methods such as picking a category based on a range of values, where each specific range corresponds to a specific category.
