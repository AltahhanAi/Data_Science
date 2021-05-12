# Basic numerical classification

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * build a nearest neighbour distance-based classifier

    *	understand the issues of overfitting and underfitting related to distance-based clustering and classification algorithms

    *	optimise hyper parameters k in a k-NN classification.


**In unit 2, we covered an important and basic classification technique that depends on decision trees. Remember that classification is an important and pervasive type of data mining problem, where we have in our dataset a categorical label for each instance in the dataset. The set of labels are called the classes, and when we are presented with an instance x, we are required to guess/predict what the label for the presented instance is.**


<figure role="group">
  <img src="../images/DS_IMG031.png" alt="A schematic illustration of classification." />
  <figcaption><strong>Figure 1.</strong> A schematic illustration of classification.</figcaption>
</figure>

In this lesson and subsequent lessons, we will continue our coverage of classification techniques but will move to a different type of classification technique that takes a different approach. This is a more numerical classification that needs all the attributes to be numerical or be converted before the technique can be applied. Similar to clustering, the technique we cover here needs to utilise the proximity measures that we have talked about in the previous lesson. In later units, we cover more classification techniques that are not lazy techniques and that have similar requirement of numerical treatment.
