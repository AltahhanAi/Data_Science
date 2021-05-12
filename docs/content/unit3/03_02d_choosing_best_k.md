# Choosing best k

**Similar to K-means, the best k can be specified as a hyper parameter by trying different values and employing the elbow method.**

You can refer back to optimisation of a hyper parameters via cross validation if the dataset is small. In the case of a relatively big dataset, we need to split the dataset into testing training and look at when the performance between testing and training forks. Effectively, the training error would be decreasing while the testing error is increasing. In this case, the value of k can be backtracked to where the two measures diverged from each other. We increase k gradually and stop increasing k when we find out that kNN test errors start to increase.

!!! abstract "Exercise"
    See the following <a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-18754875_4" target="_blank">Excel sheet</a> that implements a distance-weighted voting $k$NN algorithm. Try adjusting the values in the dataset (in the sheet) to see the effect on the algorithm’s different steps.
