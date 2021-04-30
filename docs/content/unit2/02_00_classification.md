# Classification

<mark>In this unit you will</mark>

!!! success "Learning outcomes:"
	After completing this unit you should be able to:

    * <mark>outcome 1</mark>

    * <mark>outcome 2</mark>

    * <mark>outcome 3</mark>


In a classification scenario, we have a supervised learning problem with labels. The labels themselves are a categorical data, i.e. they have a set of confined values, be it numerical or categorical that we can enumerate through. We call these values the classes, and our task is to take an input data point and place it in the area or under one of the classes. This type of task is pervasive in all aspects of data mining, and we will find ourselves facing it in one form or the other in several settings. One distinction we need to make here is between classification and clustering. While classification uses predefined classes with names, clustering is unsupervised learning, in which data points are placed in clusters based on their intrinsic properties and the similarities between them.  

There are numerous techniques for classification, but in this unit we will focus on two of the simplest and most widely used: Decision Trees and Linear Models. In later modules such Machine Learning, Deep Learning and Text Analytics, you will come across several other techniques, building on the ideas that we develop here. Ok let us get started…

Figure 1 shows a schematic representation of a classification model. As it can be seen, the classification model maps an input x with an output y. the input is a set of attributes for one record and the output is the predicted class of the record. The record can be any object or entity represented in our dataset as one record.  

<figure role="group">
  <img src="../images/DS_IMG031.png" alt="Test image." />
  <figcaption><strong>Figure 1.</strong> A schematic illustration of classification.</figcaption>
</figure>

In the next few lessons we cover common classification techniques. Namely we will cover decision trees, K-nearest neighbours and linear models. These are among the simplest, widely used and effective techniques. We will see also their strengths and limitations and when they are most suitable for which type of problems. We will demonstrate important concepts along the side of the techniques such as applying data preparation and performance measure.

Many problems are actually classification problem in disguise and our mission as data scientist is to spot them and be able to engineer the associated data if necessary to make it suitable for the types of techniques that we want to apply.
