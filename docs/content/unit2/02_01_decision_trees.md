# Decision trees

<mark>In this lesson you will</mark>

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * <mark>outcome 1</mark>

    * <mark>outcome 2</mark>

**Let us assume that we have the following binary tree structure:**

<figure role="group">
  <img src="../images/DS_IMG008.png" alt="Test image." />
  <figcaption><strong>Figure 1.1.</strong> A binary tree structure.</figcaption>
</figure>

In this diagram you can see that we have used leaves (round edged rectangles) and nodes (ovals). The nodes represent the conditions that need to be checked, while the leaves represent a decision to isolate or not to isolate. This is a binary tree since each condition has two results only (either yes or no). In other words each node can have two children only representing the two possible results of the condition that the node represents. The tree structure represents a binary decision tree. Note that DTs do not necessarily need to be binary, each node can have any number of children. However, a condition of a tree structure is for each node to have one and only one parent (if not then it is just a graph- a tree is special type of a graph). This condition helps the tree to satisfy several guarantees that simplify the inference and its inception process. Ok, so you might be asking now, what do you mean inference? We talk about it in the next section.

##Decision trees inference
An **inference** is the process of using a model (such as a DT) in order to **infer** what is the class (label in general) of a given record (case, or data point). The process of creating a DT is called **training the tree**. We will see an algorithm that shows us to build a DT automatically form the data. But first we will have a look at how we can infer the class of a case from the DT.

Let us assume that we have been given the following new case and we want our DT to tell the concerned person whether to isolate or not.

| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Jasmin| No       |  Yes                             | **?**  |

The DT then checks first if the person has symptoms.

<figure role="group">
  <img src="../images/DS_IMG009.png" alt="Test image." />
  <figcaption><strong>Figure 1.2.</strong> Inference in a decision tree, following left hand side branch of level 0.</figcaption>
</figure>

Since the person has no symptoms, the inference process will go to the next node to check if they have been in contact with a positive case recently:

<figure role="group">
  <img src="../images/DS_IMG010.png" alt="Test image." />
  <figcaption><strong>Figure 1.3.</strong> Inference in a decision tree, following left hand side branch of level 1.</figcaption>
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
  <img src="../images/DS_IMG011.png" alt="Test image." />
  <figcaption><strong>Figure 1.4.</strong> Inference in a decision tree, following right hand side branch of level 0.</figcaption>
</figure>

| Name  | Symptoms | Close contact to a positive case | Isolate|
|-------|----------|----------------------------------|--------|
| Joe   | Yes      |  No                              | **Yes**|

As we can see the DT inference did not use all the available data checks since it can reach a conclusion without having to check for the second condition.

Those conditions constitute the features or the attribute for our data. The cases are the records, while decision is the label or the class of the case.

###Example of an invalid DT

The following structure is not valid decision tree since we have several possibilities of the same ‘Has a Job’ condition:

<figure role="group">
  <img src="../images/DS_IMG012.png" alt="Test image." />
  <figcaption><strong>Figure 1.5.</strong> Invalid decision tree for loan eligibility. This decision tree is invalid due to multiple branches of the same value for ‘Has a Job’ feature.</figcaption>
</figure>

We can alter it to make a valid decision tree as follows:

<figure role="group">
  <img src="../images/DS_IMG013.png" alt="Test image." />
  <figcaption><strong>Figure 1.6.</strong> A valid decision tree for loan eligibility.</figcaption>
</figure>

So we can see that the graph becomes a valid DT by creating a new condition (attribute); the band of the income of the mortgage applicant which has three possible cases (band1, band2 and band3). So this DT is not binary. This is fine and we can conduct inference on this tree as before. Note that the tree is not binary but the **class** is binary {eligible, not eligible}. Given the following case:

Name |  Home Owner  | Has a job  |Income band | Eligible
-----|--------------|------------|------------|---------
Emma |  Yes         | Yes        | Band 1     |     ?

The decision tree path will be shown in red all at once (but bear in mind that it will be conducted in stages as we showed earlier).

<figure role="group">
  <img src="../images/DS_IMG014.png" alt="Test image." />
  <figcaption><strong>Figure 1.7.</strong> Loan eligibility decision tree, showing the inference path.</figcaption>
</figure>

Given that both cases of High and Low Bands are eligible then we can further simplify the tree as follows:

<figure role="group">
  <img src="../images/DS_IMG015.png" alt="Test image." />
  <figcaption><strong>Figure 1.8.</strong> A better decision tree structure for loan eligibility.</figcaption>
</figure>


###Exercise

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




##Summary

<mark>**In this lesson you have....**

In the next lesson you will ...</mark>
