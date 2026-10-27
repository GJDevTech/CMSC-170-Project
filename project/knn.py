
"""
!!! DO NOT REMOVE !!!

knn classification
assigned to calvin and fae
reference: lab 06 knn

dataset
- use new-dataset/training.csv for training
- use new-dataset/testing.csv for testing
- use diagnosis as target label (B or M)
- discard id and diagnosis from input features
- use all the 30 numerical features

implementation
- implement knn from scratch following lab 06
- use k = 5 and p = 2 (euclidean)
- apply min-max scaling to all features
- calculate min and max values from training data only
- use the same min and max values for testing data
- handle features with zero range to avoid division by zero
- calculate distances between each test record and all training records
- select the k nearest neighbors based on distance
- predict the diagnosis using majority voting

evaluation
- use the existing 75/25 stratified split
- calculate accuracy and malignant recall
- display the confusion matrix (TP, TN, FP, FN)
- treat malignant (M) as the positive class
- compare results with naive bayes

notes
- do not modify or resplit the provided datasets
- implement the algorithm and metrics from scratch
- do not use scikit or other libraries
"""
