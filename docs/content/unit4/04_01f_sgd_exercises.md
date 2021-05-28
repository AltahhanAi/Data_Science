#Evaluating a regression model via R2 and the RMSE

Besides using the loss mean of squared errors to drive the learning process, we can utilise it to measure the performance of the learned mode. The problem with this approach is that the squares give inflated estimation of the error. To deal with this, we can use the root mean squared error (RMSE) instead to measure the model performance. Please distinguish between measure performance and the loss function, although they are related we often use easy to differentiate continues function as the loss, while for regression performance evaluation we can use any whole-measure to give a point evaluation for the model. RMSE is given as:

$$
R M S E=\left(\overline{J^{2}}\right)^{0.5}
$$

Another possibility, when we want to be able to compare the performance of the model in different datasets, (or between different models and dataset) is to use the coefficient of determination of $R^2$ metric. $R^2$ is defined as:

$$
R^{2}=1-\frac{\sum_{n=1}^{N}\left(t_{n}-y_{n}\right)^{2}}{\sum_{n=1}^{N}\left(t_{n}-\bar{t}\right)^{2}} \mid
$$

where $\bar{t}$ is the mean of the target. The numerator is referred to as SSR or SSE and is our usual sum of prediction squared errors, while denominator is denoted as SST (sum of squared total) and is the sum of squared deviation from the mean (which is the variance of the target times the size of the data under consideration). It represents the sum of squared errors for a model that uses the mean of the target data as the prediction for all data points.

$R^{2}$ gives an indication of the extent to which our model is better than just using the mean of the target to predict all data points’ values. The closer the value is to 1, the better the model is. The closer the value is to 0.5 the less useful the model is. If the metric is less than 0.5 then that indicates that it is worse than guessing as per the mean. The range of $R^{2}$ is [0,1].


!!! abstract "Exercise"

    See the following Jupyter notebook to compare the performance of vanilla and vectorised SGD, the code is written in pure python code using numpy.

    - Download exercise (.ipynb):   <a href="../exercises/Exercise3_SGD_Algorithm_LinearRegression.ipynb" target="_blank" download>Exercise 3 </a>

##Simple linear regression models: lesson summary

In this lesson you have learnt about simple linear regression models, and their coefficient that maps into weights. You have understood the generalisation from one dimension of input into multi-dimensional input space. You have seen how to utilise loss function effectively in order to solve the linear regression problem exactly via least squares, and approximately via the gradient descent. In addition, you have learnt how to address sequential learning problems by utilising the ideas of mini-batch learning.
