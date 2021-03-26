# Decision tree induction (training)

In this lesson you will learn about...
!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

	* <mark>outcome 1</mark>
	* <mark>outcome 2</mark>
	* <mark>outcome 3</mark>
	* <mark>outcome 4</mark>

In the below dataset, the possible values for the features and the class are as follows:  

Screen size = {6, 7, 8}, Makes calls = {Yes, No}, Class = {Phone, Tablet}

Device | Screen size  | Makes calls | Classification
-------|--------------|-------------|---------------
G1     |     6 inches | Yes         |Phone
S1     |     6 inches | Yes         |Phone
A1     |    7 inches  | Yes         |Phone
G2     |    7 inches  | No          |Tablet
S2     |   7 inches   | No          |Tablet
A2     |   8 inches   | Yes         |Tablet

Intuitively, the ‘screen size’ feature does not have an easy binary decision since in this dataset we have phones that are 6 and 7 inches and we have tablets that are also 7 or 8 inches. On the other hand, ‘makes calls’ feature is more regular, **all** phones ‘make calls’ while **most** tablets do not ‘make calls’. This suggests that if we were to make decisions about a devise class being a phone or a tablet, we should first look at the ‘makes calls’ attribute and if it is ‘no’ then it is a ‘tablet’ if it is ‘yes’ then it is most likely a ‘phone’. In this second instance we would then need to check its ‘screen size’ if it is = 8 then it is a tablet, if it is not then it is a phone.

So how we can create an algorithm that does this type of decisions for us? We need an algorithm that can strategically pick the more promising features first and then develop the tree based on that. The algorithm that we will talk about is called CART (Classification and Regression Tree) and we will show it in action in the following steps.

##Step 1 of CART algorithm

**To be able to develop this algorithm we need to measure how promising a feature (with a specific value) is as a partition for our dataset. Think about the above dataset. The ‘makes calls’ feature allowed us to split the data two ways with a low impurity.**

<figure role="group">
  <img src="../images/DS_IMG016.png" alt="Test image." />
  <figcaption><strong>Figure 2.1.</strong> Illustration of step 1 of CART algorithm for tablet vs phone dataset. Left split based on ‘makes calls’ feature, right split based on ‘screen size=8’ .</figcaption>
</figure>

To quantify the quality of each split, we use the **Gini Impurity** of each **node data**. The Gini impurity describes how pure or mixed the data labels in a node. The purer the data the closer Gini is to 0, while the more mixed the data in the node the closer Gini is to 0.5. We will look at how to calculate the Gini Impurity a bit later. For now I want you to assume that you know how to calculate it.

###Information gain
The **Information Gain of a split** refers to how much information we gain by choosing one of the possible splits. If the decision tree is binary, then each feature is a possible split. If it is not binary, then each pair of a feature with a value is represented as (feature, value), and will constitute a possible split.  

If we would like to decide which of the above two splits is better, then we just calculate the information gain and we choose the one that has the highest information gain.

<mark>Equation</mark>
$Information Gain = Impurity of parent – weighted average impurity of children$


Info Gain('makes calls') $=0.5 –( 26×0+46×0.375)=0.25$


Info Gain('screen size=8') $=0.5 –( 56×0.48+16×0)=0.1$

This shows that the first split based on the ‘makes calls’ feature is better since we gain form information by using it and we will intuitively move toward more pure leaves.

###Gini impurity

**It is time now to see how we calculate the Gini impurity. We take the probability of one label in the node and we multiply it with the probability of the other label (or sum of other labels probabilities if we have multi-class dataset). And we do that again for the second label and so on.**

So, to see this measure in action in a binary class dataset (like our dataset), let us take the probability of an item being a tablet in our dataset, which we denote as
$p(Tab)=3/6=0.5$. This is because we have 6 items in total 3 of them are tablets. The same applies for the phones where we have $p(Pho)=3/6=0.5$. Hence, the Gini Impurity of the dataset before any split is:  

$Gini Impurity(set)=p(Tab)[1-p(Tab)]+p(Pho)[1-p(Pho)]=p(Tab)+p(Pho)- [p^2 (Tab)+p^2 (Pho)]$

However we have $p(Tab)+p(Pho)=1$, therefore:

$Gini Impurity(set) = 1- [p^2 (Tab)+p^2 (Pho)]$

Below we will see the calculations of the Gini Impurity for the subsets of the nodes (the items that belong to each node). We start always from the whole dataset and then the data will be distributed based on the type of question that we ask in the node. The summary of how we evaluate the two splits is in <mark>Figure ().</mark>

<mark>equations table</mark>

$Impurity(■(G_1@P)■(S_1@P)■(A_1@P)■(G_2@T)■(S_2@T)■(A_2@T))=1-[〖(□(3/6))〗^2+〖(□(3/6))〗^2 ]=0.5$

<figure role="group">
  <img src="../images/DS_IMG017.png" alt="Test image." />
  <figcaption><strong>Figure 2.2.</strong> Illustration of step 1 of CART algorithm for tablet vs phone dataset, showing information gain calculations. Split based on ‘makes calls’ feature.</figcaption>
</figure>

$Impurity(■(G_2@T)■(S_2@T))=1-[〖(□(2/2))〗^2+〖(□0)〗^2 ]=0
Impurity(■(G_1@P)■(S_1@P)■(A_1@P)■(A_2@T))=1-[〖(□(1/3))〗^2+〖(□(2/3))〗^2 ]=0.375
Gain('makes calls')  =0.5 –( □(2/6)×0+□(4/6)×0.375)$


<figure role="group">
  <img src="../images/DS_IMG018.png" alt="Test image." />
  <figcaption><strong>Figure 2.3.</strong> Illustration of step 1 of CART algorithm for tablet vs phone dataset, showing information gain calculations. Split based on ‘screen size=8’.</figcaption>
</figure>

$Impurity(■(G_1@P)■(S_1@P)■(A_1@P)■(G_2@T)■(S_2@T))=1-[(□(3/5))^2+(□(2/5))^2 ]=0.48
Impurity(■(A_2@T))=1-[〖(□(1/1))〗^2+〖(□0)〗^2 ]=0
Gain('screen size=8')  =0.5 –( □(5/6)×0.48+□(1/6)×0)=0.1$

Figures 2.2 and 2.3 above illustrate step 1 of CART algorithm for tablet vs phone dataset, showing information gain calculations. In figure 2.2 the split is based on ‘makes calls’, and in figure 2.3 the split is based on ‘screen size=8’.  Both are shown here with the full information gain calculations needed to decide which split to choose, and in this case we can see that 'makes calls' wins.

The algorithm will go ahead and calculate the information gain for another two splits possibilities, these are ‘screen size=6’ and ‘screen size=7’. We have not shown these, so if you’d like to give it a try yourself you can do the calculations now. The results should be in favour of ‘makes calls’ split.

!!! note
     The Gini impurity deals with classes in a binary manner (one versus the rest). So, if we have multi-class dataset then we would simply enumerate through the different classes and deal with each label in the node as the target class and the rest as misclassification. For example, if we have say 3 classes, {‘tablet’, ‘phone’, ‘portable PC’} the Gini impurity will be:

      <mark>equations</mark>

Tan et al (2020) use Entropy and a slightly different algorithm for building the tree called Hunt’s Algorithm, here we use the CART algorithm which is widely used for DT.  

##Step 2 of CART algorithm

**Next, the CART algorithm will convert the branch on the left of the ‘makes calls’ into a leaf since it’s a pure node (all of its data point are of class ‘tablet’). The right hand side node is a mixture of 3 ‘phones’ and a ‘tablet’.**

<figure role="group">
  <img src="../images/DS_IMG019.png" alt="Test image." />
  <figcaption><strong>Figure 2.4.</strong> Illustration of step 2 of CART algorithm for tablet vs phone dataset. Left split based on ‘screen size=8’ feature, right split based on ‘screen size=7’.</figcaption>
</figure>

After we have chosen the ‘makes calls’ split where we have exhausted its different possibilities (the yes and no values), we move to the next feature ‘screen size’ (which happens to be the last feature that we have in our simple dataset). Since we said that there are only three values that this feature can take {6, 7, 8} we have three splits that can be done based on this feature. The algorithm will evaluate each split and we will choose the best one.

<mark>equations table</mark>

If you’d like to try it yourself now, you can calculate the information gain for ‘screen size=6’.The results should be in favour of ‘screen size=8’. The final results are summarised in figures 2.5 and 2.6.

<figure role="group">
  <img src="../images/DS_IMG020.png" alt="Test image." />
  <figcaption><strong>Figure 2.5.</strong> Illustration of step 2 of CART algorithm for tablet vs phone dataset. Split based on ‘screen size=8’ feature’.</figcaption>
</figure>

Impurity(■(G_1@P)■(S_1@P)■(A_1@P))=1-[(□(3/3))^2+(□0)^2 ]=0
Impurity(■(A_2@T))=1-[(0)^2+(□(1/1))^2 ]=0
Gain('scr size=8')  =0.375 –( □(3/4)×0+□(1/4)×0)=0.375

<figure role="group">
  <img src="../images/DS_IMG021.png" alt="Test image." />
  <figcaption><strong>Figure 2.6.</strong> Illustration of step 2 of CART algorithm for tablet vs phone dataset. Split based on ‘screen size=7’.</figcaption>
</figure>

Impurity(■(G_1@P)■(S_1@P)■(A_2@T))=1-[(□(2/3))^2+(□(1/3))^2 ]=0.44
Impurity(■(A_1@P))=1-[〖(□(1/1))〗^2+〖(□0)〗^2 ]=0
Gain('scr size=7')=0.375 –( □(3/4)×0.44+□(1/4)×0)=0.0416

Figures 2.5 and 2.6 above illustrate step 2 of CART algorithm for tablet vs phone dataset, showing information gain calculations. In figure 2.5 the split is based on ‘screen size=8’, and in figure 2.6 the split is based on ‘screen size=7’.  Both are shown here with the full information gain calculations needed to decide which split to choose, and in this case we can see that ‘screen size=8’ wins.

Based on the above step, the algorithm will reach the following form:

<figure role="group">
  <img src="../images/DS_IMG022.png" alt="Test image." />
  <figcaption><strong>Figure 2.7.</strong> Final Step of CART algorithm tree induction (training).</figcaption>
</figure>

At this stage the algorithm stops since all lower levels nodes are pure and produces the following final tree which can be used for inference as we did earlier in the previous section <mark>[link].</mark>

<figure role="group">
  <img src="../images/DS_IMG023.png" alt="Test image." />
  <figcaption><strong>Figure 2.8.</strong> Final Tree structure after CART algorithm finished training for phone vs tablet dataset.</figcaption>
</figure>

!!! abstract "Exercise"
        Mimic a DT inference to deduce the class of the following devices:

				Device | Screen size  | Makes calls | Classification
				-------|--------------|-------------|---------------
				M1     | 7 inches     | Yes         |  **?**
				K1     | 8 inches     | Yes         |  **?**

##Cart algorithm

<mark>code box?</mark>

To summarise, the CART algorithm does the following:

The above box shows the pseudocode for a decision tree induction algorithm. The algorithm works by expanding the tree using the best split attribute that yields the best information gain. E is a set of data inside a node and F is the set of attributes that we can use to split the data E.

###Discretising continuous variables

Given the following dataset, we want to build a decision tree that can predict whether or not a borrower is going to default on their debt. This type of decision is important for banks to decide upon the eligibility of customers to be lent money. While our dataset is simple, the ideas can be easily expanded into a fully developed scenario for an actual bank. <mark>Intermediate</mark>

ID | Home Owner  | Marital Status | Annual Income | Defaulted Borrower
---|-------------|----------------|---------------|-------------------
1  | Yes         | Single         |  125          | No
2  | No          | Married        |  100          | No
3  | No          | Single         |  70           | No
4  | Yes         | Married        |  120          | No
5  | No          | Divorced       |  95           | Yes
6  | No          | Married        |  60           | No
7  | Yes         | Divorced       |  150          | No
8  | No          | Single         |  85           | Yes
9  | No          | Married        |  75           | No
10 | No          | Single         |  90           | Yes

So far we have only dealt with categorical features. However, as you will see in the dataset above, the ‘Annual Income’ feature is not a categorical but a continuous feature. To be able to build a decision tree for the above dataset we can discretise the ‘Annual Income’ feature (in the next section we will see how to deal with it without discretisation). One way to discretise this feature is to partition the space values into a limited set of intervals and replace any value in the dataset by the interval that it falls into. We can give the intervals names to reflect them as categorical values of the continuous feature. The intervals can be evenly distributed or can have other distributions such as a normal distribution; this depends on the underlying distribution of the feature itself. If we think that all values are equally likely then a uniform distribution is suitable and we just divide the possible values into a set of equal ranges. For our dataset we can assume that Annual Income ranges are as shown in the following table and graph:

Annual income ranges £K | Annual income Bands
------------------------|--------------------
[18, 48[                | Basic
[48, 78[                | Intermediary
[78, 108[               | Advanced
[100, 130[              | High
[130, 160[              | Top
[160, [                 | Exec

<figure role="group">
  <img src="../images/DS_IMG024.png" alt="Test image." />
  <figcaption><strong>Figure 2.9.</strong> Annual Income bands uniformly distributed, note that the width of all the ranges are £30k.</figcaption>
</figure>

However, if we think that we might have values in the middle more than on the sides, then maybe we want to have more fine ranges in the middle and more coarse ranges on the side. We can partition the values into a set of ranges according with the a variable interval width range that is inversely normally distributed:


Annual Income Ranges £K | Annual income category
------------------------|-----------------------
[18, 48[                | Basic
[48, 66[                | Intermediary L
[66, 78[                | Intermediary H
[78, 86[                | Advanced L
[86, 93[                | Advanced M
[93, 100[               | Advanced H
[100, 108[              | Advanced T
[100, 112[              | High L
[112, 130[              | High H
[130, 160[              | Top
[160, [                 | Exec

<figure role="group">
  <img src="../images/DS_IMG025.png" alt="Test image." />
  <figcaption><strong>Figure 2.10.</strong>  Annual Income categories with an inverted normal distribution.</figcaption>
</figure>

Another way to discretise is by looking into the domain of the model and how it is used. For example in the UK salaries are categorised according to tax bands as follows:


Annual income ranges £K | Annual income increment £K | Annual income category
------------------------|----------------------------|-----------------------
54,900                  |       2,300                | Basic      
57,700                  | 2,800                      | Intermediary L
61,000                  | 3,300                      | Intermediary H
65,000                  | 4,000                      | Advanced L
70,200                  | 5,200                      | Advanced M
76,800                  | 6,600                      | Advanced H
86,000                  | 9,200                      | Advanced T
98,600                  | 12,600                     | High L
121,000                 | 22,400                     | High H
175,000                 | 54,000                     | Exec

<figure role="group">
  <img src="../images/DS_IMG026.png" alt="Test image." />
  <figcaption><strong>Figure 2.11.</strong>  Top 10 UK actual annual income in 2018, the increments have reversed Pareto distribution..</figcaption>
</figure>

As can be seen, the increments take a long tailed (skewed) distribution that is not a Gaussian, but more of a reversed Pareto distribution. This is not surprising as the Pareto distribution has historically been used to describe wealth in society. The 80-29 Pareto principle is related to this distribution but is precisely realised when the alpha value is 1.16. It takes the form:

<mark>Equation</mark>$Pr(X>x)={1−(xminx)αwhen x≥xmin1  when x<xmin$

Note that the categories’ names {‘Basic, ‘Intermediary L’ Exec’} are arbitrary and could be changed to any values that suit the usage of the model (L, M, H, T stands for Low, Medium, High and Top, respectively).

##Split for continuous variables

**In the examples we have used so far we have seen how to split for features with categorical values. For ‘Screen Size’ we restricted the possibilities into 3 values, making it effectively categorical. We also discretised the ‘Annual Income’ by dealing with 3 ranges of salaries called ‘bands’, also effectively yielding it as categorical.**

However, discretisation is not always possible and can restrict the generalisation ability of our models. What we want to be able to do is to allow a continuous variable (such as ‘Annual Income’) to take any value for continuous variables, and we decide from the data what would be the best value to split the data according to. To understand the approach let us look at a tangible example. To deal with a split of continuous features we simply look into its values inside the available dataset. Remember we have in theory an infinite number of values so we cannot try them all!.  

ID | Home owner| Marital status | Annual income | Defaulted borrower | **Possible splits for annual income**|
---|-----------|----------------|---------------|--------------------|--------------------------------------|
6  | No        | Married        | 60            | No                 |                                      |
3  | No        | Single         | 70            | No                 | **65**        
9  | No        | Married        | 75            | No                 | **72.5**
8  | No        | Single         | 85            | Yes                | **80**
10 | No        | Single         | 90            | Yes                | **87.5**
5  | No        | Divorced       | 95            | Yes                | **92.5**
2  | No        | Married        | 100           | No                 | **97.5**
4  | Yes       | Married        | 120           | No                 | **110**
1  | Yes       | Single         | 125           | No                 | **122.5**
7  | Yes       | Divorced       | 150           | No                 | **172.5**

Table (): Borrowers dataset with possible splits for the Annual Income feature

1. We need to sort the dataset according to this feature, and we take the split values to be in-between the feature values in the dataset.  

2. We take the in-between values instead of the values themselves because we do not want to make any of the dataset records a boundary case. We do not need to worry about the first and last values since they cannot be a split condition otherwise they yield the feature ineffective- all data is greater than the first value and smaller than the last values. So if we have N records in our dataset (N=10 in the Borrowers dataset), we try N-1 in-between splits. See Table <mark>()</mark> above for the possible splits for annual income after sorting the dataset according to ‘Annual Income’.

3. Then we now try to split according to each in-between value, and we calculate the Gini index and information gain for the results. We compare between all the information gain of the different splits and we take the split that maximises the information gain. Note that all the calculations that we talked about in the previous section apply. Since the original data Gini is not going to vary, we can simply take the split that minimises the Gini index since Information Gain = Gini for parent – Gini for the split. See <mark>table ()</mark> below for the different Information Gain and Gini Index calculations.

<mark>Table (7)</mark>: Borrowers dataset with information gain calculations for possible splits for the Annual Income feature. The datasheet with all the formulas is available as an <mark>Excel file here.</mark>

Figure 2.12 shows the advantage of a test condition for a continuous attributes, the branching of the tree is much simpler and will lead to a more elegant and less cultured and easy to interpret tree.

<figure role="group">
  <img src="../images/DS_IMG032.png" alt="Test image." />
  <figcaption><strong>Figure 2.12.</strong>  Test condition for a continuous attribute.</figcaption>
</figure>

##Other types of impurity measurements

**There are other impurity measures such as the entropy or the misclassification error. Figure 2.13 below shows the behaviour of these three impurity measures as per two probabilities of two classes.**

We only show one probability on the x axis because the other is just the complement of p, i.e. 1-p. As you can see, when both probabilities of the two classes are close to 0.5 the impurity is maximal. When either is close to the 1 (the other 1-p would be close to 0) the impurity is minimised. The figure shows that the max of the Gini and misclassification error is 0.5 while the max for the entropy is 1.

<figure role="group">
  <img src="../images/DS_IMG033.png" alt="Test image." />
  <figcaption><strong>Figure 2.13.</strong>  Comparison of different impurity measures.</figcaption>
</figure>

In fact, these are all valid and you can use any. Albeit an important element of decision tree induction which must be used, changing the impurity measure between these three measures has a limited effect on the tree structure. In fact, although they vary in range, they produce consistent decision trees. What matters more in the context of decision trees is the use of pruning and pre-pruning. Therefore we will only briefly discuss them here, but you should try to familiarise yourself with these other types of impurity measures from Tan et al (2020), particularly entropy which has applications in a wide range of disciplines.  

The entropy is a measure of chaos in a system. It is also used as a measure of information- in fact, information theory depends heavily on it. In this lesson however, we will concentrate on it as a measure of chaos or surprise. If the set of events or items have a probability peak, i.e. a subset of those items have high probability, then the system is less chaotic and the entropy is small. On the other hand, if the events or items have similar probabilities, without a clear winner, then the system is harder to predict and its chaos or entropy is maximal.  

This is reflected in <mark>figure 2.13</mark> above, where we can see that when the two classes have a probability of 0.5 (remember if p=0.5 then 1-p=0.5) then the entropy is maximal =1. It fades away when one of the classes has high probability and the higher the probability the lower the entropy until it reaches 0, when the probability of either classes is 1 (same for one of the classes probability is close to 0 the other would be close to 1). There is always symmetry in all of those impurity when dealing with a binary class problem, but when it is multi-class this is not guaranteed. Below we contrast Gini and entropy to gain understanding of both.

For Class 1 with probability $p$, we want to make sure that:

1. When the probability $p$ is low, the $Entropy$ is low. Hence, we simply include $p$ in $Entropy$ formula at the same time.

2. When the probability $p$ is high, the $Entropy$ is low. Hence, we include the term $−logp$ in the $Entropy$ formula.

Note that $logp≤0$ because $p≤1$. Hence $−logp≥0$.

Note that $−logp$ is monotonically decreasing function.

Note also that the base of $log$ is normally 2 but any can do as long as we are consistent. The behaviour of $−logp$ for class C1 and $−log(p′)$ for class C2 can be seen below. When the probability increases $−logp$ decreases but it is still positive (to be precise it is non-negative). Note that $−log(p′)$ is monotonically increasing function with respect to $p$ and is non-negative as well.

<figure role="group">
  <img src="../images/DS_IMG034.png" alt="Test image." />
  <figcaption><strong>Figure 2.14.</strong>  Behaviour of the term $−plog p$ which is the entropy for class C1. Note that C1 has a probability $p$ and the figure shows how the entropy of C1 is varying with the probability $p$..</figcaption>
</figure>

To take into account both of the points above, the entropy for class C1 will be written as $−plog p$, which has a behaviour that is described in the left hand side of figure 2.15 below. In addition, since we have two classes then we also need a similar term for the second class C2. Given that C2 has a probability $p′=1−p$, its entropy is $(1−p)log(1−p)$. The behaviour of this term is shown in the right hand side of the figure below.

<figure role="group">
  <img src="../images/DS_IMG035.png" alt="Test image." />
  <figcaption><strong>Figure 2.15.</strong>  Left: The entropy for class C1= $−plog p$. C1 has a probability $p$, the figure shows how the entropy of C1 varies with the probability $p$. Right: The entropy for class C2= $−p'log p′$. C2 has a probability $p′=1−p$, the figure shows how the entropy of C2 varies with the probability $p$.</figcaption>
</figure>

We can finally define the entropy as:

$Entropy=−plogp−p′logp′$

$Entropy=−plogp−(1−p)log(1−p)$

Its behaviour is shown figure 2.16 below.

<figure role="group">
  <img src="../images/DS_IMG036.png" alt="Test image." />
  <figcaption><strong>Figure 2.16.</strong>  Left: The entropy of both classes C1 and C2 who have probabilities $p$ and $p′=1−p$, respectively. Right: The different components of the entropy fit together.</figcaption>
</figure>

Note that we are talking about two classes (events) not two probability distributions. In the case of two probability distributions we use cross-entropy which is outside the scope of this discussion. In general if we have more than $K$ classes, then:

$Entropy=∑Ki=1pilogpi$

###Comparison of the entropy with Gini Index

The same idea applies for the Gini index, but it is less complex.

1. When the probability $p$ is low, the $Gini$ is low. Hence, we simply include $p$ in $Gini$ formula. <mark>At the same time</mark>

2. When the probability $p$ is high, the $Gini$ is low. Hence, we include the term $1−p$ in the $Gini$ formula.

To take into account both of the points above, the Gini index should include the term $p(1−p)$. Its behaviour is shown in <mark>figure 24</mark> below.

<figure role="group">
  <img src="../images/DS_IMG037.png" alt="Test image." />
  <figcaption><strong>Figure 2.17.</strong> The behaviour of the term $(1−p)$ with respect to class C1 which has probability $p$.</figcaption>
</figure>

<figure role="group">
  <img src="../images/DS_IMG038.png" alt="Test image." />
  <figcaption><strong>Figure 2.18.</strong> left: The Gini impurity for class C1 = $p(1−p)$, C1 has probability $p$. Note that the term $1−p$ replaces the $−logp$ in the entropy and it is easier to calculate.</figcaption>
</figure>

Note that $1−p$ happens to be the probability of class C2 but it is not what is meant here, this becomes clearer when we consider a multi-class situation where the term $(1−p)$ is still used to calculate the impurity of C1 but the probability of C2 is likely to be different due to the involvement of other classes. This coincidence makes the left and right hand sides identical for the binary classes problems. Note that the term has a max of 0.5*0.5=0.25.

In addition, since we have two classes then we need also similar term for the second class. Given that its probability is $p′$

$Gini=p(1−p)+p′(1−p')$

In the case of Gini impurity it is helpful to realise that $p+p′=1$ hence:

$Gini=p(1−p)+p′(1−p′)=(p+p′)−(p2+p′2)=1−(p2+p′2)$

Its behaviour is shown in figure 2.19 below:

<figure role="group">
  <img src="../images/DS_IMG039.png" alt="Test image." />
  <figcaption><strong>Figure 2.19.</strong> Left: The Gini impurity for two classes C1 and C2 with probabilities $p$ and $p′=1−p$ respectively. Note that the Gini impurity has a max of 0.25+0.25=0.5. Right: The different components of the Gini impurity fit together..</figcaption>
</figure>

In general if we have more than $K$ classes, then:

$Gini=∑Ki=1pi(1−pi)=1−∑Ki=1p2i$

Figure 2.20 below summarises all of the terms included in both the entropy and Gini. As we have said earlier, both produce consistent trees and have a similar behaviour albeit having different ranges.

<figure role="group">
  <img src="../images/DS_IMG040.png" alt="Test image." />
  <figcaption><strong>Figure 2.20.</strong> The behaviour of the entropy Gini with respect to both class 1 which has probability	$p$ and class 2 which has probability $1−p$.</figcaption>
</figure>

Note that the colours are representative of the terms involved in the calculation of both measures. The Gini is represented as red since on $p$ and $1-p$ are involved in its calculations, while the entropy is represented as magenta since all the four terms in blue and red are involved in its calculations (red + blue=magenta).  

Finally the classification error is given as:  

$Classificaiton error=1−max(pi)$

The behaviour of all of the three impurity measures have been already shown in <mark>Figure (20)</mark>

##Summary

<mark>**In this lesson you have**</mark>
