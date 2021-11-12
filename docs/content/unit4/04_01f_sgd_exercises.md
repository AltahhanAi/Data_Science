#Evaluating a regression model via R2 and the RMSE

Besides using the loss mean of squared errors to drive the learning process, we can utilise it to measure the performance of the learned model. The problem with this approach is that the squares give inflated estimation of the error. To deal with this, we can use the root mean sum of squared errors (RMSE) instead to measure the model performance. Please distinguish between performance measure and the loss function, although they are related we often use easy to differentiate continuous function as the loss, while for regression performance evaluation we can use any whole-measure to give a point evaluation for the model. RMSE is given as:

$$
R M S E=(M S E)^{0.5}
$$

Another possibility, when we want to be able to compare the performance of the model in different datasets, (or between different models and dataset) is to use the coefficient of determination of $R^2$ metric. $R^2$ is defined as:

$$
R^{2}=1-\frac{\sum_{n=1}^{N}\left(t_{n}-y_{n}\right)^{2}}{\sum_{n=1}^{N}\left(t_{n}-\bar{t}\right)^{2}}
$$

where $\bar{t}$ is the mean of the target. The numerator is referred to as SSR or SSE and is our usual sum of squared errors, while denominator, denoted as SST (sum of squared total), is the sum of squared deviation from the mean (which is the variance of the target times the size of the data under consideration). It represents the sum of squared errors for a base model, where the base model uses the mean of the targets as the prediction for any data points.

$R^{2}$ gives an indication of the extent to which our model is better than just using the mean of the target to predict all data points’ values. The closer the value is to 1, the better the model is. If the metric is less than 0 then that indicates that it is worse than guessing as per the mean.


!!! abstract "Exercise"

    See the following Jupyter notebook to compare the performance of vanilla and vectorised SGD, the code is written in pure python code using numpy.

    - Download exercise (.ipynb):   <a href="../exercises/Exercise3_SGD_Algorithm_LinearRegression.ipynb" target="_blank" download>Exercise 3 </a>

Please watch the following three videos for a comprehensive example that covers the different concepts of this lesson.

You can download the spreadsheets Abdulrahman refers to in the videos below:

<a href="../files/Comprehensive_Example_from_Scratch.xlsx" target="_blank" download>Comprehensive example from scratch (.xlsx) </a>

<a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-22591869_4" target="_blank">Comprehensive example with feature mapping (.xlsx) </a>

<iframe title="video 6 (Example of a linear regression model, part 1) " width="450" height="300" frameborder="0" scrolling="auto" marginheight="0" marginwidth="0" src="https://mymedia.leeds.ac.uk/Mediasite/Play/031ebb9609ad4e4882c3f0ab85781e3e1d" allowfullscreen msallowfullscreen allow="fullscreen"></iframe>

<a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-22914403_4" target="_blank">Download transcript (PDF).</a>

 <iframe title="video 7 (Example of a linear regression model, part 2) " width="450" height="300" frameborder="0" scrolling="auto" marginheight="0" marginwidth="0" src="https://mymedia.leeds.ac.uk/Mediasite/Play/8d4658ebca5e45619f40502216b597ea1d" allowfullscreen msallowfullscreen allow="fullscreen"></iframe>

 <a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-22914404_4" target="_blank">Download transcript (PDF).</a>

<iframe title="video 8 (Example of a linear regression model, part 3" width="450" height="300" frameborder="0" scrolling="auto" marginheight="0" marginwidth="0" src="https://mymedia.leeds.ac.uk/Mediasite/Play/1d7e63fe54454aaaa93a7669c613a7e41d" allowfullscreen msallowfullscreen allow="fullscreen"></iframe>

<a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-22914409_4" target="_blank">Download transcript (PDF).</a>  

##Simple linear regression models: lesson summary

In this lesson you have learnt about simple linear regression models, and their coefficient that maps into weights. You have understood the generalisation from one dimension of input into multi-dimensional input space. You have seen how to utilise loss function effectively in order to solve the linear regression problem exactly via least squares, and approximately via the gradient descent. In addition, you have learnt how to address sequential learning problems by utilising the ideas of mini-batch learning.
