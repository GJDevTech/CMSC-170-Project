
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

"""
example
training_rows = [
    {
        "diagnosis": "B",
        "radius_mean": 12.5,
        "texture_mean": 18.2,
        # ... 28 more features
    },
"""
def calculate_priors(training_rows):

    # 1. count records under each class
    b = 0
    m = 0

    for record in training_rows:        # for each training record
        diagnosis = record["diagnosis"]   # read diagnosis

        # check if it's B or M, increment accordingly
        if diagnosis == "M":
            m+=1
        elif diagnosis == "B":
            b+=1
        else:
            print("Error reading record")

    
    # 2. for each class, calculate priors
    priors = {}
    total = len(training_rows)

    priors["B"] = b / total
    priors["M"] = m / total

    # 3. return priors and class counts as dict
    class_counts = {"B": b, "M": m}

    return priors, class_counts


def calculate_likelihoods(training_rows, feature_names, class_counts):

    # 1. init. every possible combination of class, feature, and category to 0
    counts = {}  
    for class_label in CLASSES:          # B or M
        counts[class_label] = {}

        for feature in feature_names:    # for every feature 
            counts[class_label][feature] = {}

            for category in CATEGORIES:  # for every category
                counts[class_label][feature][category] = 0

    # 2. count occurrences in training data
    for record in training_rows:                         # for each categorized training record
        class_label = record["diagnosis"]                # read diagnosis

        for feature in feature_names:                    # for every feature 
            category = record[feature]                   # read category of that feature
            counts[class_label][feature][category] += 1  # update matching count

    # 3. calculate likelihoods
    likelihoods = {}
    for class_label in CLASSES:                   # for each class
        class_count = class_counts[class_label]   # get number of training records belonging to that class
        likelihoods[class_label] = {}

        for feature in feature_names:             # for every feature
            likelihoods[class_label][feature] = {}

            for category in CATEGORIES:           # for each category
                
                # apply P(category|class) with Laplace smoothing a = 1
                likelihoods[class_label][feature][category] = (counts[class_label][feature][category] + 1 ) / (class_count + 3)

    # 4. return likelihoods
    return likelihoods

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