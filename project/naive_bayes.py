
"""
!!! DO NOT REMOVE !!!

modified naive bayes classification
assigned to jericho and gj
reference: lab 04 nb classifier

dataset
- use new-dataset/training.csv for training
- use new-dataset/testing.csv for testing
- use diagnosis as target label (B or M)
- discard id and diagnosis from input features
- use all the 30 numerical features

implementation
- implement categorical naive bayes from scratch
- convert continuous features into three categories using discretization
- use low, medium, and high as the categories
- calculate the 33rd and 67th percentile boundaries from training data
- apply the same category boundaries to testing data
- assign values at or below the first boundary to low
- assign values at or below the second boundary to medium
- assign remaining values to high
- calculate prior probabilities for B and M
- calculate feature likelihoods for each category and class
- apply laplace smoothing with alpha = 1
- use (count + 1) / (class count + 3) for feature likelihoods
- calculate class scores using log probabilities
- predict the diagnosis using the class with the higher score

evaluation
- use the existing 75/25 stratified split
- calculate accuracy and malignant recall
- display the confusion matrix (TP, TN, FP, FN)
- treat malignant (M) as the positive class
- compare results with knn

notes
- do not modify or resplit the provided datasets
- implement the algorithm and metrics from scratch
- do not use scikit or other libraries
"""
