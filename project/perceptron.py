"""
!!! DO NOT REMOVE !!!

perceptron classification
assigned to ALL
reference: lab 08 perceptron (P3. Introduction to Machine Learning)

dataset
- use new-dataset/training.csv for training
- use new-dataset/testing.csv for testing
- use diagnosis as the target label (B or M)
- discard id and diagnosis from the input features
- use all 30 numerical features

implementation
- implement one single-layer perceptron from scratch
- apply min-max scaling using minimum and maximum from training data only
- use the same training minimum and maximum for testing data
- convert M to +1 and B to -1 for the perceptron learning rule
- initialize all feature weights to 0 and the bias to 0.5
- compute activation = sum(weight * feature value) + bias
- predict +1 if activation >= 0, otherwise predict -1
- if prediction differs from desired output, update weights and bias
- use learning rate r = 0.1
- perform full passes (epochs) over training data in original row order
- stop if an epoch has no updates, or after at most 100 epochs
- convert +1 back to M and -1 back to B for evaluation

evaluation
- use the existing 75/25 stratified training/testing split
- calculate accuracy and malignant recall from scratch
- display confusion counts (TP, TN, FP, FN)
- treat malignant (M) as the positive class
- compare results with knn and nb

notes
- do not modify or resplit the supplied datasets
- do not train or fit scaling parameters on testing data
- implement the algorithm and evaluation metrics from scratch
- do not use scikit-learn, pandas, or other high-level ML libraries
- initial bias, learning rate, zero-boundary rule, and epoch cap are project settings; slides use bias 0.5 and r 0.1
"""

import csv

# constants
TRAINING_FILE = "new-dataset/training.csv"
TESTING_FILE = "new-dataset/testing.csv"
CLASSES = ("B", "M")
LEARNING_RATE = 0.1
INITIAL_BIAS = 0.5
MAX_EPOCHS = 100


# PART A: DATA PREP
# Assigned to: Fae


def load_dataset(file_path):
    # open the file and read each row as a dictionary
    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        # feature names are every column except id and diagnosis
        feature_names = []
        for column in reader.fieldnames:
            if column != "id" and column != "diagnosis":
                feature_names.append(column)

        # keep diagnosis as text, convert every feature to float
        rows = []
        for record in reader:
            row = {"diagnosis": record["diagnosis"]}

            for feature in feature_names:
                row[feature] = float(record[feature])

            rows.append(row)

    return rows, feature_names


def calculate_min_max(training_rows, feature_names):
    min_max = {}

    for feature in feature_names:
        # collect this feature's values from all training records
        values = []
        for row in training_rows:
            values.append(row[feature])

        # store its smallest and largest value
        min_max[feature] = (min(values), max(values))

    return min_max


def scale_value(value, minimum, maximum):
    value_range = maximum - minimum

    if value_range == 0:
        return 0.0

    return (value - minimum) / value_range


def scale_dataset(rows, feature_names, min_max):
    scaled_rows = []

    for row in rows:
        # start a new row with the same diagnosis
        scaled_row = {"diagnosis": row["diagnosis"]}

        # scale every feature using the training minimum and maximum
        for feature in feature_names:
            minimum, maximum = min_max[feature]
            scaled_row[feature] = scale_value(row[feature], minimum, maximum)

        scaled_rows.append(scaled_row)

    return scaled_rows


def encode_diagnosis(label):
    # code here
    pass


# PART B: PERCEPTRON COMPUTATION
# Assigned to: Calvin


def initialize_parameters(feature_names, initial_bias):
    # code here
    pass


def calculate_activation(record, feature_names, weights, bias):
    # code here
    pass


def predict_output(activation):
    # code here
    pass


# PART C: PERCEPTRON TRAINING
# Assigned to: Jericho

# Update perceptron's weights and bias using learning rule
def update_parameters(record, feature_names, weights, bias, desired_output, predicted_output, learning_rate):
    
    # 1. update weights
    error = desired_output - predicted_output          # calculate error
    for feature in feature_names:                      # for every feature
        scaled_value = record[feature]                 # read that feature's scaled value from record
        change = learning_rate * error * scaled_value  # calculate change
        weights[feature] += change                     # add change to feature's weight

    # 2. update bias
    bias_change = learning_rate*error  # calculate bias change
    bias += bias_change                # add bias change to bias

    # 3. return parameters
    return weights, bias               

# Train perceptron until convergence or we reach max number of epochs
def train_perceptron(training_rows, feature_names, learning_rate, max_epochs, initial_bias):

    # check if max_epochs is at least 1 and learning_rate is within (0, 1]
    if max_epochs < 1 or not (learning_rate > 0 and learning_rate <= 1):
        raise ValueError(f"max_epochs {max_epochs} should at least be 1 and 0 < {learning_rate} <= 1.")

    # check if training rows is empty
    if not training_rows:
        raise ValueError("training_rows is empty.")

    # 1. initialize parameters
    weights, bias = initialize_parameters(feature_names, initial_bias)
    converged = False
    epochs_run = 0

    # 2. train perceptron
    while epochs_run < max_epochs:  # repeat up to max_epochs epochs

        updates_this_epoch = 0  # init. 

        for record in training_rows:        # for each training record
            label = record["diagnosis"]     # get diagnosis
            desired_output = encode_diagnosis(label)      # get desired output
            activation = calculate_activation(record, feature_names, weights, bias)  # get activation
            predicted_output = predict_output(activation)   # get predicted output

            # check if predicted output is incorrect
            if predicted_output != desired_output:
                # if not, update weights and bias
                weights, bias = update_parameters(record, feature_names, weights, bias, desired_output, predicted_output, learning_rate)
                # count number of updates in this epoch
                updates_this_epoch+=1

        # 3. record completed epochs
        epochs_run += 1

        # 4. check if no updates, stop
        if updates_this_epoch == 0:
            converged = True
            break
        # else repeat next epoch
    
    return weights, bias, epochs_run, converged


# PART D: EVALUATION AND INTEGRATION
# Assigned to: GJ


def predict_dataset(testing_rows, feature_names, weights, bias):
    # code here
    pass


def evaluate_predictions(testing_rows, predictions):
    # predictions must line up with the testing rows
    if len(testing_rows) != len(predictions):
        raise ValueError("Number of predictions does not match number of testing records.")

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # compare each actual diagnosis with its prediction
    for i in range(len(testing_rows)):
        actual = testing_rows[i]["diagnosis"]
        predicted = predictions[i]

        if actual == "M" and predicted == "M":
            TP += 1
        elif actual == "B" and predicted == "B":
            TN += 1
        elif actual == "B" and predicted == "M":
            FP += 1
        elif actual == "M" and predicted == "B":
            FN += 1
        else:
            raise ValueError("Invalid diagnosis label")

    # accuracy = (TP + TN) / total, use 0.0 if there are no records
    total = TP + TN + FP + FN
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = (TP + TN) / total

    # malignant recall = TP / (TP + FN), use 0.0 if there are no malignant records
    if TP + FN == 0:
        malignant_recall = 0.0
    else:
        malignant_recall = TP / (TP + FN)

    # store everything (same keys the Naive Bayes main uses)
    results = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "malignant_recall": malignant_recall
    }

    return results

def main():
    """
    STEPS:
    # 1. Load existing training and testing files.
    # 2. Confirm both files use the same 30 input features.
    # 3. Get scaling values from TRAINING records only.
    # 4. Apply the same training-derived scaling to BOTH sets.
    # 5. Train only on scaled_training.
    # 6. Predict test labels without changing weights or bias.
    # 7. Compare predictions against the held-out test labels.
    # 8. Display training status and the same metrics used by KNN and NB.
    """
    pass

if __name__ == "__main__":
    main()
