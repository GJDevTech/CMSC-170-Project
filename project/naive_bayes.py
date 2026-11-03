
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

# constants 
TRAINING_FILE = "new-dataset/training.csv"
TESTING_FILE = "new-dataset/testing.csv"
CLASSES = ("B", "M")
CATEGORIES = ("low", "medium", "high")


# PART A: DATA PREP
# Assigned to: GJ

def load_dataset(file_path):
    # code here
    pass


def calculate_boundaries(training_rows, feature_names):
    # code here
    pass


def categorize_value(value, lower_boundary, upper_boundary):
    # code here
    pass


def categorize_dataset(rows, feature_names, boundaries):
    # code here
    pass


# PART B: NB CLASSIFIER
# Assigned to: Jericho

def calculate_priors(training_rows):
    # code here
    pass


def calculate_likelihoods(training_rows, feature_names, class_counts):
    # code here
    pass


def predict_one(record, feature_names, priors, likelihoods):
    # code here
    pass


# PART C: EVALUATION
# Assigned to: GJ

def evaluate_predictions(testing_rows, predictions):
    # code here
    pass


# PART D: MAIN
# Assigned to: Jericho

def main():

    # load data
    training_rows, features = load_dataset(TRAINING_FILE)
    testing_rows, test_features = load_dataset(TESTING_FILE)

    # check if columns are compatible
    if features != test_features:
        raise ValueError("Training and testing features do not match.")

    # fit boundaries on training data only
    boundaries = calculate_boundaries(training_rows, features)

    # apply the same boundaries to both sets
    categorized_training = categorize_dataset(training_rows, features, boundaries)
    categorized_testing = categorize_dataset(testing_rows, features, boundaries)

    # calculate class priors
    priors, class_counts = calculate_priors(categorized_training)

    # calculate feature likelihoods
    likelihoods = calculate_likelihoods(categorized_training, features, class_counts)

    # start making predictions
    predictions = []
    for record in categorized_testing:
        prediction = predict_one(record, features, priors, likelihoods)
        predictions.append(prediction)

    # evaluate predictions
    results = evaluate_predictions(categorized_testing, predictions)

    # display results for comparison later
    print("\nNAIVE BAYES RESULTS")
    print("-------------------")
    print("Accuracy:", results["accuracy"])
    print("Malignant Recall:", results["malignant_recall"])
    print("TP:", results["TP"])
    print("TN:", results["TN"])
    print("FP:", results["FP"])
    print("FN:", results["FN"])


if __name__ == "__main__":
    main()