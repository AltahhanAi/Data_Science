# Measuring the performance of a classification model

**In this section we will study how we can measure how accurate our classification model is. The concepts and ideas will be applied to a decision tree because this is the only classification technique we have covered so far, but the concepts and ideas are equally applicable regardless of the technique. Indeed, we will be utilising these metrics in later sections and units directly without explanation.**

Measuring the effectiveness of a model is an essential skill, since we will often face problems that we can solve in several techniques. To objectively choose among them we need to compare their respective predictive models' capabilities and pick the one that suits best our problem. There is also the issue of picking the right **metric** for the problem in hand. Therefore, we will cover several metrics and we will learn which suits a specific problem, both in terms of the dataset structure and in terms of the aim of the analysis that we intend to perform in order to solve the problem in hand.

We will start by classification of a binary class problem, and we will generalise the measures for multi-class problems. However, please bear in mind that some metrics do not directly extend to a multi-class problem, and we will point this out whenever it is the case. But before seeing these metrics, we need to talk about why in the first place we might have imperfect performance.

##Measuring the performance of a binary class model

**For a binary class problem, we have two classes that an instant can belong to. It can belong to class C1 or to class C2 but cannot belong to both at the same time. In fact, the question can be posed as whether an instant belongs to a one class of concern or not.**

The classification would be effectively stating yes or 1 or + if the instant belongs to the main class of concern or stating no or 0 or – if the instant does not belong to the main class (which implicitly means it belongs to the complement of the class). We just have to be consistent in our approach. So in this context it is a binary choice, and it is useful to represent one of the classes as positive + and the other as negative –. Now, our classifier (our DT) mission is to predict whether an instant is of class + or class –. Therefore, we contrast what the classifier has **predicted** and what was the **actual class** of all instances to measure how good our classifier is. This can be done for the training set, the validation set or the testing set, all of which we have answers for (i.e. we know the classes for these sets). When we contrast the prediction against the actual class of each instant we have four possibilities:

1.	**The actual class is + and    the predicted class is + 	(True Positive-TP)**
2.	**The actual class is – while the predicted class is +	(False Positive-FP)**

3.	**The actual class is + while the predicted class is –	(False Negative-FN)**
4.	**The actual class is – and    the predicted class is –	(True Negative-TN)**

These 4 cases can be better summarised in figure 4.1 below. This is called the confusion matrix because the red boxes represent the cases confused by the prediction model, while the blue boxes represent the cases where the predictions of the model are aligned with the reality. The aim of any model is to reduce the cases in the white boxes and make them as close to 0 as possible. We can actually do a lot with these counts, and below we show several metrics that can be defined based on them.

<figure role="group">
  <img src="../images/DS_IMG062.png" alt="Test image." />
  <figcaption><strong>Figure 4.1</strong> Confusion matrix for binary classification mode .</figcaption>
</figure>

!!! info
    Be mindful that some sources present the confusion matrix using the transpose of the above matrix (with actual classes on the left and the predicted classes on top) as in figure 4.2 below. This will not change anything but the presentation. RapidMiner and Weka for example use the first form, while Tan et al (2020) uses the second form. We will adopt the first form as it is more common. Note that **FP** is called **type I error**, while **FN** is called **type II error**, accordingly it makes more sense to present the matrix in the first form.

    <figure role="group">
      <img src="../images/DS_IMG063.png" alt="Test image." />
      <figcaption><strong>Figure 4.2</strong> Confusion matrix for binary classification model presented differently.</figcaption>
    </figure>

Note that the total number of instances is the sum of all of the numbers in the boxes of the confusion matrix:

$$
𝒏 = TP+ TN+ FP+ FN
$$

This is regardless of the distribution of the correctly and incorrectly classified instances.

We define the Accuracy of a classifier as the rate of correctly classified instances out of the total number of instances:

$$
Accuracy=(TP+TN)/n
$$

On the other hand, we define the Error rate as the rate of the incorrectly classified instances out of the total number of instances:

$$
\text { Error rate }=(F P+F N) / n
$$

All classifiers try to increase its accuracy or equivalently reduce its error rate. However, in some special cases these metrics do not reflect how good the model is. For example, if the classes are not balanced, the accuracy can be misleading. We will study these cases and more suitable measures for them in unit 4.

<figure role="group">
  <img src="../images/DS_IMG064.png" alt="Test image." />
  <figcaption><strong>Figure 4.3</strong> Metrics that are related to actual classes (common names used). Left: positive actual classes related metrics. Right: negative actual classes related metrics.</figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG065.png" alt="Test image." />
  <figcaption><strong>Figure 4.4</strong> Metrics that are related to predicted classes (common names used). Left: positive predicted classes’ related metrics. Right: negative predicted classes’ related metrics.</figcaption>
</figure>

As it can be seen, all possible ways of taking the rate of either of the four values in the confusion matrix is covered and has its own properties. Of particular interest is the **precision** and the **recall** (aka positive prediction value and true positive rate, respectively). When we talk about precision, we are referring to the precision of the **prediction** of our classifier with respect to the positive class. The recall, on the other hand, measures how good our classifier in detecting the positive **actual** cases is.

If we are less concerned with false positives then we can use the hit rate (aka recall) to choose between two models. If we are diagnosing cancer for example, then misdiagnosing people as having cancer is preferred over missing those who actually have cancer (given that there would be further checking to confirm the positive cases). In this case if we have to choose between two models with the same accuracy, but with different hit rates then we choose the one with the higher hit rate. We note however that these names are a bit confusing and some simplification and uniformity is needed in order to reveal their intrinsic properties and the relationship between each other. Therefore, we propose to rename them for consistency and uniformity as in the below figures. Note that these are our own naming and some coincide with the metrics names in the literature and some do not.

<figure role="group">
  <img src="../images/DS_IMG066.png" alt="Test image." />
  <figcaption><strong>Figure 4.5</strong> Detective Metrics related to actual classes (new suggested names used for consistency. Left: performance metrics. Right: error metrics..</figcaption>
</figure>

The bar on top represents a complement of an event in a probabilistic sense. The true positive detection rate is denoted as $p\left(\right.$ detect $\left._{+}\right)$ and it represents the probability of correctly detecting positive instances by the model. Similarly, the true negative detection rate is denoted as $p\left(\right.$ detect $\left._{-}\right)$. It represents the probability of correctly detecting negative instances by the model.  

On the other hand, the false positive detection rate is denoted $p\left(\overline{\text { detect }}_{+}\right)$ and it represents the probability of incorrectly detecting positive instances by the model. While, the false negative detection rate is denoted $p\left(\overline{\text { detect }}_{-}\right)$ and represents the probability of incorrectly detecting negative instances by the model.

We can now easily verify that:

$$
\begin{array}{l}
p\left(\text { detect }_{+}\right)+p\left(\text { detect }_{+}\right)=1 \\
p\left(\text { detect }_{-}\right)+p\left(\text { detect }_{-}\right)=1
\end{array}
$$

The names are meant to reflect the inner relationship between the different metrics. To see why, we first note that
$¬ TP = FN, ¬ TN=FP$, where we use $¬$ to denote the logical not. Now, if we negate both sides of the positive detection rate equation: $\neg\left(\right.$ Detect $\left._{+}=T P /(T P+F N)\right)$ we get $\neg$ Detect $_{+}=F N /(F N+T P)=\overline{\text { Detect }}_{+}$.

Similarly, if we negate both sides of the negative detection rate equation: $\neg\left(\right.$ Detect $\left._{-}=T N /(F P+T N)\right)$ we get $\neg$ Detect $_{-}=F P /(T N+F P)=\overline{\text { Detect }}$.

In other words, $\overline{\text { Detect }}_{+}$ $\overline{\text { Detect }}_{-}$ represents the model inability to detect the positive and negative instances, respectively.

Similar argument is used for the prediction related metrics. The true positive prediction value is denoted as
$p\left(\right.$ predict $\left._{+}\right)$ and represents the probability of correctly predicting positive instances by the model. The true negative prediction value is denoted as $p($ predict_ $)$ and represents the probability of correctly predicting negative instances by the model. On the other hand, the false positive prediction value is denoted $p\left(\overline{\text { predict }}_{+}\right)$, it represents the probability of incorrectly predicting positive instances by the model. While, the false negative prediction value is denoted $p\left(\overline{\text { predict }}_{-}\right)$, it represents the probability of incorrectly predicting negative instances by the model.  

We can now easily verify that:

$$
\begin{array}{l}
p\left(\text { predict }_{+}\right)+p\left(\overline{\text { predict }}_{+}\right)=1 \\
p\left(\text { predict }_{-}\right)+p\left(\overline{\text { predict }}_{-}\right)=1
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG067.png" alt="Test image." />
  <figcaption><strong>Figure 4.6</strong> Predictive Metrics related to predicted classes (new suggested names used for consistency). Left: performance related predictive metrics. Right: error related predictive metrics. </figcaption>
</figure>

Negation on the prediction metrics yields similar but not quite the same relationship as for the detection metrics. This is because if we negate both sides of the positive prediction value equation: $\neg\left(\right.$ Predict $\left._{+}=T P /(T P+F P)\right)$ we get $\neg$ Predict $_{+}=F N /(F N+T N)=\overline{\text { Predict }}_{-}$.

Similarly, if we negate both sides of the negative prediction value equation: $\neg\left(\right.$ Predict $\left._{-}=T N /(T N+F N)\right)$ we get $\neg$ Predict $_{-}=F P /(F P+T P)=\overline{\text { Predict }}_{+}$. Note that the negation here changed also the prediction metric form positive to negative.  

Note that we used capital initial for the metrics to express them as a rate, while we uses small letter when we place them in the context of probabilities, so for example $\overline{\text { Predict }}_{-}=p\left(\overline{\text { Predict }}_{-}\right),$ Predict $_{+}=p\left(\right.$ predict $\left._{+}\right)$ and so on.

###Which type of metric is more important?

Note that predictive metrics are concerned with the model ability to predict or guess the correct class of the instances, while  detective metrics are concerned with the model ability to detect or recognise the correct class of the instances. In particular, detective metrics are discriminative ones by nature since we can interpret it as if we are giving the model a set of positive only (or negative only) instances without revealing the class to the model and ask the model if it can tell us which one of these are positive (or which ones are negative). The answer should be all positive (or all negative) and the more the model diverges from this answer the less discriminative it is. On the other hand,  predictive metrics are speculative by nature since we can interpret them as if we are giving the model a set of mixed class instances and we ask it to come up with a guess or prediction on the class. So, predictive metrics may appear (due to its name) to pertain more for a prediction problem. However this is not the case; both the detective and predictive metrics give different insights of the quality of the classification model. The question is, can we come up with metrics that encompasses both types of measures, the answer is yes.

###Holistic metrics

**Holistic metrics** are those metrics that look at both horizontal and vertical views and involve both positive and negative classes. These are better metrics for model comparison of most problems when we are concerned with an overall good performance without a particular preference of guessing ability or discrimination ability of the model. Among these holistic metrics are the F1 score and Matthew Correlation Coefficient. The F1 score is defined as the harmonic mean of the precision and recall. Harmonic means differs from the usual arithmetic mean. The harmonic mean for $n$ numbers $x_{i}$ is defined as the reciprocal of the arithmetic mean of the reciprocals of the given numbers: $\left(\frac{\sum_{i=1}^{k} x_{i}^{-1}}{k}\right)^{-1}$.

Therefore, for two numbers it is defined as $\left(\frac{1 / x_{1}+1 / x_{2}}{2}\right)^{-1}=\frac{2 x_{1} x_{2}}{x_{1}+x_{2}}$

Therefore, F1 score is given as $F 1$ score $=\frac{2 \text { PPV.TPR }}{P P V+T P R}=\frac{\text { 2TP }}{2TP \text { (1) }+\text { FP }+\text { FN }}$

!!! Note
    Note that $𝑨𝒄𝒄𝒖𝒓𝒂𝒄𝒚=(𝐓𝐏+𝐓𝐍)/𝒏 =(𝐓𝐏+𝐓𝐍)/(𝐓𝐏+ 𝐓𝐍+ 𝐅𝐏+ 𝐅𝐍)$. If we compare this with the F1 score formula, we realise that we can obtain F1 score directly from the accuracy formula, by replacing the term $TN$ with $TP$. In other words, we can view the F1 score from another perspective as being a measure of overall accuracy for the true positive predicted cases only (no true negative).

<figure role="group">
  <img src="../images/DS_IMG068.png" alt="Test image." />
  <figcaption><strong>Figure 4.7</strong> Holistic Metrics that are comprehensively involving both predicted and actual classes. Left: F1 score. Right: Matthew Correlation Coefficient. Both are suitable for any binary class problem even when the classes’ counts are imbalanced. (i.e. when the number of instances form one class – normally the positive class – are much smaller than number of instances from the second class – normally the negative class).. </figcaption>
</figure>

!!! Note
    The Matthew correlation coefficient (MCC) is defined as:

    $$
    M C C=\frac{\text { TN. TP }-\text { FP.FN }}{\sqrt{(T P+F P)(T N+F N)(T P+F N)(T N+F P)}}
    $$

<figure role="group">
  <img src="../images/DS_IMG069.png" alt="Test image." />
  <figcaption><strong>Figure 4.8</strong> Holistic Metrics that are comprehensively involving both predicted and actual classes. Left: Accuracy. Right: Error rate. Both performs poorly when the classes count is imbalanced (i.e. when the number of instances form one class – normally the positive – are much smaller than number of instances from the second class – normally the negative class). Nevertheless, these are the basic metrics that several algorithms use. </figcaption>
</figure>

The range of MCC is between -1 and 1. -1 represents total disagreement between the model predictions and the actual classes, 1 represents total agreement between the predicted and actual classes and 0 means no correlation, i.e. the model is not better than a random guess.

In terms of comparison, we can meaningfully compare as follows:

<figure role="group">
  <img src="../images/DS_IMG194b.png" alt="Test image." />
  <figcaption><strong>Figure 4.9</strong> </figcaption>
</figure>

As can be seen we either compare using the left-hand side for performance or the right-hand side for errors. We do not need to use both, and we must be aware not to mingle the left with right when we compare different models' performance.

###Examples of metrics for a classifier

Let us now have a look at some examples. Let us assume that we trained a decision tree classifier and it gave us the confusion matrix that can be seen below. We have stated all the metrics that we have covered so far and we demonstrated the calculations in a separate box underneath. Later we might just state the metrics values without the calculations.  

By comparing the Accuracy (or F1 score or MCC) we accordingly prefer classifier 4.

<figure role="group">
  <img src="../images/DS_IMG070.png" alt="Test image." />
  <figcaption><strong>Figure 4.10</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{10}{10+40}=0.2 & \text { Detect }_{-}=\frac{40}{40+10}=0.8 \\
\text { Predict }_{+}=\frac{10}{10+10}=0.5 & \text { Predict }_{-}=\frac{40}{40+40}=0.5 \\
\text { Accuracy }=\frac{10+40}{100}=0.5 \quad F 1=\frac{2 \times 10}{2 \times 10+40+10}=0.286 \\
M C C=\frac{10 \times 40-10 \times 40}{\sqrt{(10+10)(10+40)(40+40)(40+10)}}=0.0
\end{array}
$$


<figure role="group">
  <img src="../images/DS_IMG071.png" alt="Test image." />
  <figcaption><strong>Figure 4.11</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{25}{25+25}=0.5 & \text { Detect }_{-}=\frac{25}{25+25}=0.5 \\
\text { Predict }_{+}=\frac{25}{25+25}=0.5 & \text { Predict }_{-}=\frac{25}{25+25}=0.5 \\
\text { Accuracy }=\frac{25+25}{100}=0.5 & F 1=\frac{2 \times 25}{2 \times 25+25+25}=0.5 \\
M C C=\frac{25 \times 25-25 \times 25}{\sqrt{(25+25)(25+25)(25+25)(25+25)}}=0.0
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG072.png" alt="Test image." />
  <figcaption><strong>Figure 4.12</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{40}{40+10}=0.8 & \text { Detect }_{-}=\frac{10}{10+40}=0.2 \\
\text { Predict }_{+}=\frac{40}{40+40}=0.5 & \text { Predict }_{-}=\frac{10}{10+10}=0.5 \\
\text { Accuracy }=\frac{40+10}{100}=0.5 \quad F 1=\frac{2 \times 40}{2 \times 40+40+10}=0.615 \\
M C C=\frac{40 \times 10-40 \times 10}{\sqrt{(40+40)(40+10)(10+40)(10+10)}}=0.0
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG073.png" alt="Test image." />
  <figcaption><strong>Figure 4.13</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{40}{40+10}=0.8 & \text { Detect }_{-}=\frac{40}{40+10}=0.8 \\
\text { Predict }_{+}=\frac{40}{40+10}=0.8 & \text { Predict }_{-}=\frac{40}{40+10}=0.8 \\
\text { Accuracy }=\frac{40+40}{100}=0.8 \quad F 1=\frac{2 \times 40}{2 \times 40+40+10}=0.615 \\
M C C=\frac{40 \times 40-10 \times 10}{\sqrt{(40+10)(40+10)(40+10)(40+10)}}=0.6
\end{array}
$$

###Examples of metrics on classifiers comparison

Let us now assume that we trained a further two classifiers that produced the following two confusion matrices. Comparing the classifiers we can see how the different metrics react to the changes in the way the instances has been classified. In particular we can see that F1 score reflect a balanced estimation of the quality of the classifier, while accuracy can be quite optimistic in its estimation of the quality of the classifier. MCC is the more reserved of holistic metrics and it tends to be more pessimistic in its estimation.

<figure role="group">
  <img src="../images/DS_IMG074.png" alt="Test image." />
  <figcaption><strong>Figure 4.14</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{rr}
\text { Detect }_{+}=\frac{50}{50+50}=0.5 & \text { Detect }_{-}=\frac{99}{99+1}=0.99 \\
\text { Predict }_{+}=\frac{50}{50+1}=0.98 \quad \text { Predict }_{-}=\frac{99}{99+50}=0.664 \\
\text { Accuracy }=\frac{50+99}{200}=0.745 \quad F 1=\frac{2 \times 50}{2 \times 50+50+1}=0.662 \\
M C C=\frac{50 \times 99-50 \times 1}{\sqrt{(50+1)(99+50)(50+50)(99+1)}}=0.56
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG075.png" alt="Test image." />
  <figcaption><strong>Figure 4.15</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{90}{90+10}=0.9 & \text { Detect }_{-}=\frac{99}{99+1}=0.99 \\
\text { Predict }_{+}=\frac{90}{90+1}=0.989 \quad \text { Predict }_{-}=\frac{99}{99+10}=0.908 \\
\text { Accuracy }=\frac{90+99}{200}=0.945 \quad F 1=\frac{2 \times 90}{2 \times 90+10+1}=0.942 \\
M C C=\frac{90 \times 99-10 \times 1}{\sqrt{(90+1)(99+10)(90+10)(99+1)}}=0.893
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG076.png" alt="Test image." />
  <figcaption><strong>Figure 4.16</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{99}{99+1}=0.99 & \text { Detect }_{-}=\frac{99}{99+1}=0.99 \\
\text { Predict }_{+}=\frac{99}{99+1}=0.99 & \text { Predict }_{-}=\frac{99}{99+1}=0.99 \\
\text { Accuracy }=\frac{99+99}{200}=0.99 & F 1=\frac{2 \times 99}{2 \times 99+1+1}=0.99 \\
M C C=\frac{99 \times 99-1 \times 1}{\sqrt{(99+1)(99+1)(99+1)(99+1)}}=0.98
\end{array}
$$

Accordingly we prefer classifier 3 since we are comparing on the same dataset. In particular, the reader needs to be careful on what constitutes a statistically significant difference of two different models.

###Classes imbalance and metrics

Let us now see how these metrics react to an increase in one of the classes, i.e. when the problem has imbalanced classes issue.

<figure role="group">
  <img src="../images/DS_IMG078.png" alt="Test image." />
  <figcaption><strong>Figure 4.17</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{c}
\text { Detect }_{+}=\frac{90}{90+10}=0.9 \mid \text { Detect }_{-}=\frac{990}{990+10}=0.99 \\
\text { Predict }_{+}=\frac{90}{90+10}=0.9 \quad \text { Predict }_{-}=\frac{990}{990+10}=0.99 \\
\text { Accuracy }=\frac{90+990}{1100}=0.981 \quad F 1=\frac{2 \times 90}{2 \times 90+10+10}=0.9 \\
M C C=\frac{90 \times 990-10 \times 10}{\sqrt{(90+10)(990+10)(90+10)(990+10)}}=0.89
\end{array}
$$


<figure role="group">
  <img src="../images/DS_IMG077.png" alt="Test image." />
  <figcaption><strong>Figure 4.18</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{50}{50+50}=0.5 & \text { Detect }_{-}=\frac{990}{990+10}=0.99 \\
\text { Predict }_{+}=\frac{50}{50+10}=0.833 \quad \text { Predict }_{-}=\frac{990}{990+50}=0.951 \\
\text { Accuracy }=\frac{50+990}{1100}=0.945 \quad F 1=\frac{2 \times 50}{2 \times 50+50+10}=0.625 \\
M C C=\frac{50 \times 990-50 \times 10}{\sqrt{(50+10)(990+50)(50+50)(990+10)}}=0.62
\end{array}
$$

Ok now let us see how the metrics react when the problem has a rare class. i.e. it is a severe imbalanced classes problem.

<figure role="group">
  <img src="../images/DS_IMG079.png" alt="Test image." />
  <figcaption><strong>Figure 4.19</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{50}{50+50}=0.5 \quad \text { Detect }_{-}=\frac{9900}{9900+100}=0.99 \\
\text { Predict }_{+}=\frac{50}{50+100}=0.333 \quad \text { Predict }_{-}=\frac{9900}{9900+50}=0.995 \\
\text { Accuracy }=\frac{50+9900}{10100}=0.985 \quad F 1=\frac{2 \times 50}{2 \times 50+50+100}=0.4 \\
M C C=\frac{50 \times 9900-50 \times 100}{\sqrt{(50+100)(9900+50)(50+50)(9900+100)}}=0.4
\end{array}
$$

<figure role="group">
  <img src="../images/DS_IMG080.png" alt="Test image." />
  <figcaption><strong>Figure 4.20</strong> Caption. See calculations below. </figcaption>
</figure>

$$
\begin{array}{cc}
\text { Detect }_{+}=\frac{90}{90+10}=0.9 & \text { Detect }_{-}=\frac{9900}{9900+100}=0.99 \\
\text { Predict }_{+}=\frac{90}{90+100}=0.473 \quad \text { Predict }_{-}=\frac{9900}{9900+10}=0.999 \\
\text { Accuracy }=\frac{90+9900}{10100}=0.989 \quad F 1=\frac{2 \times 90}{2 \times 90+10+100}=0.62 \\
M C C=\frac{90 \times 9900-10 \times 100}{\sqrt{(90+100)(9900+10)(90+10)(9900+100)}}=0.648
\end{array}
$$

##Measuring the performance of a multi-class models

Note that F1 score only applies to binary class problems. If the problem is multi-class we can use other metrics, including accuracy, recall (positive detection rate) and precision (positive prediction value). We can also treat the classes as one vs. the rest fashion (yielding the problem into a binary class problem) and obtain the F1 score for each class separately. Below we show the confusion matrix for the IRIS dataset.  For more information about the IRIS dataset read the following passage extracted from SKLearn description for the dataset.

!!! abstract "Iris plants dataset"
    **Data Set Characteristics:**

    * Number of Instances: 150 (50 in each of three classes)
    * Number of Attributes: 4 numeric, predictive attributes and the class
    * Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica

    **Summary Statistics:**

    | Attribute     |  Min | Max | Mean |  SD  | Class Correlation |
    ----------------|------|-----|------|------|-------------------|
    | sepal length: |  4.3 | 7.9 | 5.84 | 0.83 |   0.7826          |
    | sepal width:  |  2.0 | 4.4 | 3.05 | 0.43 |  -0.4194          |
    | petal length: |  1.0 | 6.9 | 3.76 | 1.76 |   0.9490  (high!) |
    | petal width:  |  0.1 | 2.5 | 1.20 | 0.76 |   0.9565  (high!) |
    - Missing Attribute Values: None
    - Class Distribution: 33.3% for each of 3 classes.
    - Creator: R.A. Fisher
    - Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    - Date: July, 1988

    **Description:**

    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken from Fisher's paper. Note that it's the same as in R, but not as in the UCI Machine Learning Repository, which has two wrong data points.
    This is perhaps the best known database to be found in the pattern recognition literature.  Fisher's paper is a classic in the field and is referenced frequently to this day.  (See Duda & Hart, for example.)  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

Note that the true labels are placed horizontally while the prediction is vertically. This is opposite to what we have used before, but as we said earlier it should not matter as long as we are vigilant about it.

<figure role="group">
  <img src="../images/DS_IMG081.png" alt="Test image." />
  <figcaption><strong>Figure 4.21</strong> Confusion matrix without normalisation. </figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG082.png" alt="Test image." />
  <figcaption><strong>Figure 4.22</strong> Confusion matrix normalised. </figcaption>
</figure>

Different sources uses these two formatting as well. On the right also you can see the same confusion matrix after normalisation. We normalise by dividing each entry by the sum along the **true label axis**. Below you will see an example that clarifies this.

<figure role="group">
  <img src="../images/DS_IMG083.png" alt="Test image." />
  <figcaption><strong>Figure 4.23</strong> Confusion matrix without normalisation. </figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG084.png" alt="Test image." />
  <figcaption><strong>Figure 4.24</strong> Confusion matrix normalised. </figcaption>
</figure>

For multi-class problems, the recall or pr(detect $\left._{\text {class }}\right)$ can be defined in terms of averaged sum of true instances of each class. To demonstrate how, let us look into the above confusion matrix. The recall for each class separately give us the following:  

$$
\operatorname{pr}\left(\text { detect }_{\text {setosa }}\right)=\frac{13}{13}, \operatorname{pr}\left(\text { detect }_{\text {versicolor }}\right)=\frac{15}{18}, \text { pr }\left(\text { detect }_{\text {virginica }}\right)=\frac{6}{7}
$$

###Holistic metric for multi-class: $pr(detect)$ aka recall score

Which yields 1.0,0.83 and 0.86 for the classes 'setosa' 'versicolor' 'virginica', respectively. Now, the recall for the above model can be calculated in several ways, some of them are:

1. Macro $p r($ detect $)=\frac{p r\left(\text { detect }_{\text {setosa }}\right)+p r\left(\text { detect }_{\text {versicolor }}\right)+p r\left(\text { detect }_{\text {virginica }}\right)}{3}=0.8968$

2. Weighted $n=13+18+7=38$. This is not 150 because the above confusion matrix is for a testing set not the total dataset. The weights for the classes are: $w_{\text {setosa }}=13 / 38 \quad w_{\text {versicolor }}=18 / 38, w_{\text {virginica }}=7 / 38$. Hence, the balanced accuracy is: $p r($ detect $)=\frac{13 p r\left(\text { detect }_{\text {setosa }}\right)+18 p r\left(\text { detect }_{\text {versicolor }}\right)+7 p r\left(\text { detect }_{\text {virginica }}\right)}{38}=0.8947$. Note that we can calculate the weighted balanced accuracy by just taking the counts of $p r($ detect $)=\frac{\# \text { True }(\text { setosa })+\# \text { True }(\text { versicolor })+\# \text { True }(\text { virginica })}{38}=\frac{13+18+6}{38}=0.8947$ which means we simply sum the diagonal of the confusion matrix and divide by the total number.

Here we would like to point out that sometimes the above score is called the balanced accuracy metric. It is defined for binary class problem as:

<figure role="group">
  <img src="../images/DS_IMG085.png" alt="Test image." />
  <figcaption><strong>Figure 4.25</strong> Caption. </figcaption>
</figure>
<mark> caption</mark>

$$
Balanced Accuracy =\frac{1}{2} (\frac{TP}{TP+FN}+\frac{TN}{TN+FP}) =\frac{1}{2} \boldsymbol{p} \boldsymbol{r}\left(\boldsymbol{d e t e c} \boldsymbol{t}_{+}\right)+\frac{1}{2} \boldsymbol{p} \boldsymbol{r}\left(\boldsymbol{d e t e c} \boldsymbol{t}_{-}\right)
$$

This metric is the macro average of the $pr(detect)$ so there is nothing new here really except that we are taking the average of both detection rates. A weighted average version can be defined as we showed earlier.

###Holistic metric for multi-class: $pr(predict)$ aka precision score

All calculations for $pr(predict)$ (aka precision) extends naturally similar to what we did for the $pr(detect)$ (aka recall). As before we can calculate the precision for each class as follows:

$$
\operatorname{pr}\left(\text { predict }_{\text {setosa }}\right)=13 / 13, \text { pr }\left(\text { predict }_{\text {versicolor }}\right)=15 / 16, \text { pr }\left(\text { predict }_{\text {virginica }}\right)=6 / 9
$$

1. Macro $\operatorname{pr}($ predict $)=\frac{ \text { pr(predict } \left._{\text {setosa }}\right)+\text { pr }\left(\text { predict }_{\text {versicolor }}\right)+\text { pr }\left(\text { predict }_{\text {virginica }}\right)}{3}=0.868$

2. Weighted $n=13+18+7=38$. The weights for the classes are: $w_{\text {setosa }}=13 / 38 w_{\text {versicolor }}=18 / 38, w_{\text {virginica }}=7 / 38$.

Hence, the balanced accuracy is: $\operatorname{pr}($ predict $)=\frac{13 \text { pr }\left(\text { predict }_{\text {setosa }}\right)+18 \text { pr }\left(\text { predict }_{\text {versicolor }}\right)+\text { 7pr }\left(\text { predict }_{\text {virginica }}\right)}{38}=0.9089$

Note here that we cannot use the counts on the diagonal of the confusion matrix as in $pr(detect)$

!!! abstract "Exercise"
    Try to calculate the balanced accuracy for the above problems and compare it with the accuracy. See if makes any difference.

!!! abstract "Exercise"    
    Research into extending the F score into a multi-class case and calculate it for the above example.
