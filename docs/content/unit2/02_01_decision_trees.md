# Decision trees

In this lesson we will see how to build a decision tree using CART induction algorithm. We will also gain an insight into using impurity measures, including Gini impurity, entropy, and misclassification error. You will use these techniques to come up with a powerful model that will have the ability to classify records which we don’t have the label for. We call these unseen records or unlabelled records.

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    - understand the intricate details of the induction process

    - explain how the impurity measures work in combination with the splitting procedure

    - understand the differences between splitting a continuous variable, and splitting categorical variables

    - carry out deduction using a decision tree and induction to build the tree.


**Let us assume that we have the following binary tree structure:**

<figure role="group">
  <img src="../images/DS_IMG008.png" alt="Diagram of a decision tree (DT) with a binary structure. Each condition node in this decision tree structure has two results only (either 'yes' or 'no')." />
  <figcaption><strong>Figure 2.2.</strong> Diagram of a decision tree (DT) with a binary structure. Each condition node in this decision tree structure has two results only (either 'yes' or 'no').</figcaption>
</figure>

In Figure 2.2 you can see that we have used leaves (round edged rectangles) and nodes (ovals). The nodes represent the conditions that need to be checked, while the leaves represent a decision to isolate or not to isolate. This is a binary tree since each condition has two results only (either yes or no). In other words each node can have two children only representing the two possible results of the condition that the node represents. The tree structure represents a binary decision tree. Note that decision trees (DTs) do not necessarily need to be binary - each node can have any number of children. However, a condition of a tree structure is for each node to have one, and only one parent (if not then it is just a graph - a tree is special type of a graph). This condition helps the tree to satisfy several guarantees that simplify the inference and its inception process. Ok, so you may now be wondering what is meant by inference? We will talk about it in the next section.

##Decision trees inference

An **inference** is the process of using a model (such as a DT) in order to **infer** what is the class (label in general) of a given record (case, or data point). The process of creating a DT is called **training the tree**. We will see an algorithm that shows us how to build a DT automatically from the data, but first we will look at how we can infer the class of a case from the DT.

Let us assume that we have been given the following new case and we want our DT to tell the concerned person whether to isolate or not.

| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Jasmin| No       |  Yes                             | **?**  |

The DT then checks first if the person has symptoms.

<figure role="group">
  <img src="../images/DS_IMG009.png" alt="Diagram showing level 0 of a decision tree (DT). The DT checks the answer ('yes' or 'no') to the condition node ('symptoms?'). The answer in this example is 'no'." />
  <figcaption><strong>Figure 2.3.</strong> Inference in a decision tree, following left hand side branch of level 0.</figcaption>
</figure>

Since the person has no symptoms, the inference process will go to the next node to check if they have been in contact with a positive case recently:

<figure role="group">
  <img src="../images/DS_IMG010.png" alt="Diagram showing level 1 of the decision tree (DT). The DT uses the inference process to check the next condition node ('close contact to a positive case?'). " />
  <figcaption><strong>Figure 2.4.</strong> Inference in a decision tree, following left hand side branch of level 1.</figcaption>
</figure>


| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Jasmin| No       |  Yes                             | **Yes**|

So as we can see the path that has been taken by the tree is specified with a red colour, while the decision is in pink. Note that the same process can be performed live by an interactive system that is directly interacting with a user. However we will not be concerned with this ability in this unit, in fact the majority of what we cover in our module will assume that the data has been already collected unless otherwise stated (for example if the data is sequential or coming from a time series). Let’s look at another example:

| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Joe   | Yes      |  No                              | **?**  |

The inference process for the DT will look like the following:

<figure role="group">
  <img src="../images/DS_IMG011.png" alt="Diagram showing level 0 of a decision tree (DT). The DT checks the answer ('yes' or 'no') to the condition node ('symptoms?'). The answer in this example is 'yes'. The inference process leads to decision ('isolate')." />
  <figcaption><strong>Figure 2.5.</strong> Inference in a decision tree, following right hand side branch of level 0.</figcaption>
</figure>

| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Joe   | Yes      |  No                              | **Yes**|

As we can see the DT inference did not use all the available data checks since it can reach a conclusion without having to check for the second condition.

Those conditions constitute the features or the attribute for our data. The cases are the records, while decision is the label or the class of the case.

###Example of an invalid DT

The following structure is not a valid decision tree since we have several possibilities of the same ‘Has a Job’ condition:

<figure role="group">
  <img src="../images/DS_IMG012.png" alt="Diagram of an invalid decision tree (DT) for loan eligibility. The DT is unable to conduct inference due to multiple answer branches of the same value ('yes') and multiple for a single condition node ('has a job')." />
  <figcaption><strong>Figure 2.6.</strong> Invalid decision tree for loan eligibility. This decision tree is invalid due to multiple branches of the same value for ‘Has a Job’ feature.</figcaption>
</figure>

We can alter it to make a valid decision tree as follows:

<figure role="group">
  <img src="../images/DS_IMG013.png" alt="Diagram of a valid decision tree (DT) for loan eligibility. This DT adds a new condition node ('income band') which then allows inference to be conducted at each level towards a binary set of decisions ('eligible' or 'not eligible')." />
  <figcaption><strong>Figure 2.7.</strong> A valid decision tree for loan eligibility.</figcaption>
</figure>

So we can see that the graph becomes a valid DT by creating a new condition (attribute); the band of the income of the mortgage applicant which has three possible cases (band1, band2 and band3). So this DT is not binary. This is fine and we can conduct inference on this tree as before. Note that the tree is not binary but the **class** is binary {eligible, not eligible}. Given the following case:

Name |  Home Owner  | Has a job  |Income band | Eligible
-----|--------------|------------|------------|---------
Emma |  Yes         | Yes        | Band 1     |     ?

The decision tree path will be shown in red all at once (but bear in mind that it will be conducted in stages as we showed earlier).

<figure role="group">
  <img src="../images/DS_IMG014.png" alt="Diagram of a decision tree for loan eligibility showing the inference path (in red)." />
  <figcaption><strong>Figure 2.8.</strong> Loan eligibility decision tree, showing the inference path.</figcaption>
</figure>

Given that both cases of High and Low Bands are eligible then we can further simplify the tree as follows:

<figure role="group">
  <img src="../images/DS_IMG015.png" alt="Diagram of a further-simplified decision tree for loan eligibility." />
  <figcaption><strong>Figure 2.9.</strong> A better decision tree structure for loan eligibility.</figcaption>
</figure>


!!! abstract "Exercise"

		Given the following cases, show the path the inference process will take on the above DT:

		Name |  Home Owner  | Has a job  |Income band | Eligible
		-----|--------------|------------|------------|---------
		Jeff |  Yes         | No         | Band 1     |     ?

		Name |  Home Owner  | Has a job  |Income band | Eligible
		-----|--------------|------------|------------|---------
		Steve|  Yes         | Yes        | Band 2     |     ?

		Name |  Home Owner  | Has a job  |Income band | Eligible
		-----|--------------|------------|------------|---------
		Mitch|  No          | Yes        | Band 1     |     ?
