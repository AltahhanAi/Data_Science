# Measuring the performance of a classification model

**In this section we will study how we can measure how accurate our classification model is. The concepts and ideas will be applied to a decision tree because this is the only classification technique we have covered so far, but the concepts and ideas are equally applicable regardless of the technique. Indeed, we will be utilising these metrics in later sections and units directly without explanation.**

Measuring the effectiveness of a model is an essential skill, since we will often face problems that we can solve in several techniques. To objectively choose among them we need to compare their respective predictive models' capabilities and pick the one that suits best our problem. There is also the issue of picking the right **metric** for the problem in hand. Therefore, we will cover several metrics and we will learn which suits a specific problem, both in terms of the dataset structure and in terms of the aim of the analysis that we intend to perform in order to solve the problem in hand.

We will start by classification of a binary class problem, and we will generalise the measures for multi-class problems. However, please bear in mind that some metrics do not directly extend to a multi-class problem, and we will point this out whenever it is the case. But before seeing these metrics, we need to talk about why in the first place we might have imperfect performance.

##Measuring the performance of a binary class model

**For a binary class problem, we have two classes <mark>that an instant is either belongs to</mark> class C1 or to class C2 but cannot belong to both at the same time. In fact, the question can be posed as whether an instant belongs to a one class of concern or not.**

The classification would be effectively stating yes or 1 or + if the instant belongs to the main class of concern or stating no or 0 or – if the instant does not belong to the main class (which implicitly means it belongs to the complement of the class). We just have to be consistent in our approach. So in this context it is a binary choice, and it is useful to represent one of the classes as positive + and the other as negative –. Now, our classifier (our DT) mission is to predict whether an instant is of class + or class –. Therefore, we contrast what the classifier has **predicted** and what was the **actual class** of all instances to measure how good our classifier is. This can be done for the training set, the validation set or the testing set, all of which we have answers for (i.e. we know the classes for these sets). When we contrast the prediction against the actual class of each instant we have four possibilities:

<mark>predictions</mark>

These 4 cases can be better summarised in <mark>the figure</mark> below. This is called the confusion matrix because the red boxes represent the cases confused by the prediction model, while the black boxes represent the cases where the predictions of the model are aligned with the reality. The aim of any model is to reduce the cases in the red boxes and make them as close to 0 as possible. We can actually do a lot with these counts, and below we show several metrics that can be defined based on them.

<mark>Figure (): Confusion matrix for binary classification mode</mark>

!!! info
    Be mindful that some sources present the confusion matrix using the transpose of the above matrix (with actual classes on the left and the predicted classes on top) as in <mark>figure ()</mark> below. This will not change anything but the presentation. RapidMiner and Weka for example use the first form, while Tan et al (2020) uses the second form. We will adopt the first form as it is more common. Note that **FP** is called **type I error**, while **FN** is called **type II error**, accordingly it makes more sense to present the matrix in the first form.

    <mark>Figure (): Confusion matrix for binary classification model presented differently. </mark>

Note that the total number of instances is the sum of all of the numbers in the boxes of the confusion matrix:

<mark> red/black issue</mark> $𝒏 = TP+ TN+ FP+ FN$

this is regardless of the distribution of the correctly and incorrectly classified instances.

We define the Accuracy of a classifier as the rate of correctly classified instances out of the total number of instances:

$Accuracy=(TP+TN)/n$

On the other hand, we define the Error rate as the rate of the incorrectly classified instances out of the total number of instances:

𝑬𝒓𝒓𝒐𝒓 𝒓𝒂𝒕𝒆=(𝐅𝐏+𝐅𝐍)/𝒏

All classifiers try to increase its accuracy or equivalently reduce its error rate. However, in some special cases these metrics do not reflect how good the model is. For example, if the classes are not balanced, the accuracy can be misleading. We will study these cases and more suitable measures for them in unit 4.

<mark>Figure (): Metrics that are related to actual classes (common names used). Left: positive actual classes related metrics. Right: negative actual classes related metrics. </mark>

<mark>Figure (): Metrics that are related to predicted classes (common names used). Left: positive predicted classes’ related metrics. Right: negative predicted classes’ related metrics.</mark>

As it can be seen, all possible ways of taking the rate of either of the four values in the confusion matrix is covered and has its own properties. Of particular interest is the **precision** and the **recall** (aka positive prediction value and true positive rate, respectively). When we talk about precision, we are referring to the precision of the **prediction** of our classifier with respect to the positive class. The recall, on the other hand, measures how good our classifier in detecting the positive **actual** cases is.

If we are less concerned with false positives then we can use the hit rate (aka recall) to choose between two models. If we are diagnosing cancer for example, then misdiagnosing people as having cancer is preferred over missing those who actually have cancer (given that there would be further checking to confirm the positive cases). In this case if we have to choose between two models with the same accuracy, but with different hit rates then we choose the one with the higher hit rate. We note however that these names are a bit confusing and some simplification and uniformity is needed in order to reveal their intrinsic properties and the relationship between each other. Therefore, we propose to rename them for consistency and uniformity as in the below figures. Note that these are our own naming and some coincide with the metrics names in the literature and some do not.

<mark>Figure (): Detective Metrics related to actual classes (new suggested names used for consistency. Left: performance metrics. Right: error metrics. </mark>

The bar on top represents a complement of an event in a probabilistic sense. The true positive detection rate is denoted as $p(detect+)$ and it represents the probability of correctly detecting positive instances by the model. Similarly, the true negative detection rate is denoted as $p(detect−)$. It represents the probability of correctly detecting negative instances by the model.  

On the other hand, the false positive detection rate is denoted $p(detect−+)$ and it represents the probability of incorrectly detecting positive instances by the model. While, the false negative detection rate is denoted $p(detect−−)$ and represents the probability of incorrectly detecting negative instances by the model.

We can now easily verify that:

$p(detect+)+p(detect−+)=1$

$p(detect−)+p(detect−−)=1$

The names are meant to reflect the inner relationship between the different metrics. To see why, we first note that
$¬ TP = FN, ¬ TN=FP$, where we use $¬$ to denote the logical not. Now, if we negate both sides of the positive detection rate equation: $¬(Detect+=TP/(TP+FN))$ we get $¬Detect+= FN/(FN+TP)=Detect−+$.

Similarly, if we negate both sides of the negative detection rate equation: $¬(Detect−= TN/(FP+TN))$ we get $¬Detect−= FP/(TN+FP)=Detect−−$. In other words, $Detect−+ Detect−−$ represents the model inability to detect the positive and negative instances, respectively.

Similar argument is used for the prediction related metrics. The true positive prediction value is denoted as
$p(predict+)$ and represents the probability of correctly predicting positive instances by the model. The true negative prediction value is denoted as $p(predict−)$ and represents the probability of correctly predicting negative instances by the model. On the other hand, the false positive prediction value is denoted $p(predict−+)$, it represents the probability of incorrectly predicting positive instances by the model. While, the false negative prediction value is denoted $p(predict−−)$, it represents the probability of incorrectly predicting negative instances by the model.  

We can now easily verify that:

$p(predict+)+p(predict−+)=1$

$p(predict−)+p(predict−−)=1$

<mark>Figure (): Predictive Metrics related to predicted classes (new suggested names used for consistency). Left: performance related predictive metrics. Right: error related predictive metrics. </mark>

Negation on the prediction metrics yields similar but not quite the same relationship as for the detection metrics. This is because if we negate both sides of the positive prediction value equation:

$¬(Predict+=TP/(TP+FP))$
we get $¬Predict+=FN/(FN+TN)=Predict−−$

Similarly, if we negate both sides of the negative prediction value equation: $¬(Predict−=TN/(TN+FN))$
we get $¬Predict−=FP/(FP+TP)=Predict−+$. Note that the negation here changed also the prediction metric form positive to negative.  

Note that we used capital initial for the metrics to express them as a rate, while we uses small letter when we place them in the context of probabilities, so for example $Predict−−=p(Predict−−)$, $Predict+=p(predict+)$ and so on.

###Which type of metric is more important?

Note that predictive metrics are concerned with the model ability to predict or guess the correct class of the instances, while  detective metrics are concerned with the model ability to detect or recognise the correct class of the instances. In particular, detective metrics are discriminative ones by nature since we can interpret it as if we are giving the model a set of positive only (or negative only) instances without revealing the class to the model and ask the model if it can tell us which one of these are positive (or which ones are negative). The answer should be all positive (or all negative) and the more the model diverges from this answer the less discriminative it is. On the other hand,  predictive metrics are speculative by nature since we can interpret them as if we are giving the model a set of mixed class instances and we ask it to come up with a guess or prediction on the class. So, predictive metrics may appear (due to its name) to pertain more for a prediction problem. However this is not the case; both the detective and predictive metrics give different insights of the quality of the classification model. The question is, can we come up with metrics that encompasses both types of measures, the answer is yes.

###Holistic metrics

**Holistic metrics** are those metrics that look at both horizontal and vertical views and involve both positive and negative classes. These are better metrics for model comparison of most problems when we are concerned with an overall good performance without a particular preference of guessing ability or discrimination ability of the model. Among these holistic metrics are the F1 score and Matthew Correlation Coefficient. The F1 score is defined as the harmonic mean of the precision and recall. Harmonic means differs from the usual arithmetic mean. The harmonic mean for $n$ numbers $xi$ is defined as the reciprocal of the arithmetic mean of the reciprocals of the given numbers:

 $(∑ki=1x−1ik)−1$

Therefore, for two numbers it is defined as $(1/x1 + 1/x22)−1=2 x1x2x1+x2$

Therefore, F1 score is given as $F1 score=2 PPV.TPRPPV+TPR=2TP2TP+ FP+ FN$

!!! Note
    Note that $𝑨𝒄𝒄𝒖𝒓𝒂𝒄𝒚=(𝐓𝐏+𝐓𝐍)/𝒏 =(𝐓𝐏+𝐓𝐍)/(𝐓𝐏+ 𝐓𝐍+ 𝐅𝐏+ 𝐅𝐍)$. If we compare this with the F1 score formula, we realise that we can obtain F1 score directly from the accuracy formula, by replacing the term $TN$ with $TP$. In other words, we can view the F1 score from another perspective as being a measure of overall accuracy for the true positive predicted cases only (no true negative).

<mark>Figure (): Holistic Metrics that are comprehensively involving both predicted and actual classes. Left: F1 score. Right: Matthew Correlation Coefficient. Both are suitable for any binary class problem even when the classes’ counts are imbalanced. (i.e. when the number of instances form one class – normally the positive class – are much smaller than number of instances from the second class – normally the negative class).  </mark>

The Matthew correlation coefficient (MCC) is defined as  $MCC=TN.TP−FP.FN(TP+FP)(TN+FN)(TP+FN)(TN+FP)−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−√$

<mark>Figure (): Holistic Metrics that are comprehensively involving both predicted and actual classes. Left: Accuracy. Right: Error rate. Both performs poorly when the classes count is imbalanced (i.e. when the number of instances form one class – normally the positive – are much smaller than number of instances from the second class – normally the negative class). Nevertheless, these are the basic metrics that several algorithms use. </mark>

The range of MCC is between -1 and 1. -1 represents total disagreement between the model predictions and the actual classes, 1 represents total agreement between the predicted and actual classes and 0 means no correlation, i.e. the model is not better than a random guess.

In terms of comparison, we can meaningfully compare as follows:
<mark>equations table</mark>

As can be seen we either compare using the left-hand side for performance or the right-hand side for errors. We do not need to use both, and we must be aware not to mingle the left with right when we compare different models' performance.

###Examples of metrics for a classifier

Ok let us now have a look at some examples. Let us assume that we trained a decision tree classifier and it gave us the confusion matrix that can be seen below. We have stated all the metrics that we have covered so far and we demonstrated the calculations in a separate box underneath. Later we might just state the metrics values without the calculations.  

By comparing the Accuracy (or F1 score or MCC) we accordingly prefer classifier 4.

<mark> unlabeled figure</mark>

###Examples of metrics on classifiers comparison
Let us now assume that we trained a further two classifiers that produced the following two confusion matrices. Comparing the classifiers we can see how the different metrics react to the changes in the way the instances has been classified. In particular we can see that F1 score reflect a balanced estimation of the quality of the classifier, while accuracy can be quite optimistic in its estimation of the quality of the classifier. MCC is the more reserved of holistic metrics and it tends to be more pessimistic in its estimation.

<mark> unlabeled figure</mark>

Accordingly we prefer classifier 3 since we are comparing on the same dataset. Please refer to the section of model comparison <mark>[LINK]</mark> for a more detailed discussion of model comparison. In particular, the reader needs to be careful on what constitutes a statistically significant difference of two different models.

###Classes imbalance and metrics

Let us now see how these metrics react to an increase in one of the classes, i.e. when the problem has imbalanced classes issue.

<mark> unlabeled figure</mark>

Ok now let us see how the metrics react when the problem has a rare class. i.e. it is a severe imbalanced classes problem.

<mark> unlabeled figure</mark>

##Measuring the performance of a multi-class models

Note that F1 score only applies to binary class problems. If the problem is multi-class we can use other metrics, including accuracy, recall (positive detection rate) and precision (positive prediction value). We can also treat the classes as one vs. the rest fashion (yielding the problem into a binary class problem) and obtain the F1 score for each class separately. Below we show the confusion matrix for the IRIS dataset.  For more information about the IRIS dataset read the following passage extracted from SKLearn description for the dataset.

<mark> confusion matrix screenshot?</mark>

Note that the true labels are placed horizontally while the prediction is vertically. This is opposite to what we have used before, but as we said earlier it should not matter as long as we are vigilant about it.

<mark> 2x confusion matrix graphs, unlabeled </mark>

Different sources uses these two formatting as well. On the right also you can see the same confusion matrix after normalisation. We normalise by dividing each entry by the sum along the **true label axis**. Below you will see an example that clarifies this.

<mark> 2x confusion matrix graphs, unlabeled </mark>

For multi-class problems, the recall or $pr(detectclass)$ can be defined in terms of averaged sum of true instances of each class. To demonstrate how, let us look into the above confusion matrix. The recall for each class separately give us the following:  

$pr(〖detect〗_setosa )=13/13 , pr(〖detect〗_versicolor )=15/18 ,  pr(〖detect〗_virginica )=6/7$

###Holistic metric for multi-class: $pr(detect)$ aka recall score

Which yields 1.0,0.83 and 0.86 for the classes 'setosa' 'versicolor' 'virginica', respectively. Now, the recall for the above model can be calculated in several ways, some of them are:

1. Macro $pr(detect)=(pr(〖detect〗_setosa )+pr(〖detect〗_versicolor )+pr(〖detect〗_virginica ))/3=0.8968$

2. Weighted $n=13+18+7=38$ This is not 150 because the above confusion matrix is for a testing set not the total dataset. The weights for the classes are: $w_setosa=13/38  w_versicolor=18/38, w_virginica=7/38$. Hence, the balanced accuracy is: $pr(detect)=(13pr(〖detect〗_setosa )+18pr(〖detect〗_versicolor )+7pr(〖detect〗_virginica ))/38=0.8947$. Note that we can calculate the weighted balanced accuracy by just taking the counts of
$pr(detect)=(#True(setosa)+#True(versicolor)+#True(virginica))/38=(13+18+6)/38=0.8947$ which means we simply sum the diagonal of the confusion matrix and divide by the total number.

Here we would like to point out that sometimes the above score is called the balanced accuracy metric. It is defined for binary class’s problem as:

<mark> unlabeled figure</mark>

$Balanced Accuracy=1/2 (TP/(TP+FN)+TN/(TN+FP))=1/2 pr(〖detect〗_+ )+1/2 pr(〖detect〗_-)$

This metric is the macro average of the $pr(detect)$ so there is nothing new here really except that we are taking the average of both detection rates. A weighted average version can be defined as we showed earlier.

###	Holistic metric for multi-class: $pr(predict)$ aka precision score

All calculations for $pr(predict)$ (aka precision) extends naturally similar to what we did for the $pr(detect)$ (aka recall). As before we can calculate the precision for each class as follows:

$pr(〖predict〗_setosa )=13/13 , pr(〖predict〗_versicolor )=15/16 ,  pr(〖predict〗_virginica )=6/9$

1. Macro $pr(predict)=(pr(〖predict〗_setosa )+pr(〖predict〗_versicolor )+pr(〖predict〗_virginica ))/3=0.868$

2. Weighted $n=13+18+7=38$ This is not 150 because the above confusion matrix is for a testing set not the total dataset. The weights for the classes are: $w_setosa=13/38  w_versicolor=18/38, w_virginica=7/38$. Hence, the balanced accuracy is: $pr(predict)=(13pr(〖predict〗_setosa )+18pr(〖predict〗_versicolor )+7pr(〖predict〗_virginica ))/38=0.9089$

Note here that we cannot use the counts on the diagonal of the confusion matrix as in $pr(detect)$

!!! abstract "Exercise"
    Try to calculate the balanced accuracy for the above problems and compare it with the accuracy. See if makes any difference.

!!! abstract "Exercise"    
    Research into extending the F score into a multi-class case and calculate it for the above example.
