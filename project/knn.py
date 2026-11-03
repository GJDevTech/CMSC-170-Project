
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

# constants
TRAINING_FILE = "new-dataset/training.csv"
TESTING_FILE = "new-dataset/testing.csv"
CLASSES = ("B", "M")
K = 5
P = 2


# PART A: DATA PREP
# Assigned to: ?

def load_dataset(file_path):
    # code here
    pass


def calculate_min_max(training_rows, feature_names):
    # code here
    pass


def scale_value(value, minimum, maximum):
    # code here
    pass


def scale_dataset(rows, feature_names, min_max):
    # code here
    pass


# PART B: KNN CLASSIFIER
# Assigned to: ?

def calculate_distance(record_a, record_b, feature_names, p):
    # code here
    pass


def find_nearest_neighbors(training_rows, record, feature_names, k, p):
    # code here
    pass


def predict_one(neighbors):
    # code here
    pass


# PART C: EVALUATION
# Assigned to: ?

def evaluate_predictions(testing_rows, predictions):
    # code here
    pass


# PART D: MAIN
# Assigned to: ?

def main():
    # code here
    pass


if __name__ == "__main__":
    main()
