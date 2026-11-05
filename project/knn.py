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

import csv

# constants
TRAINING_FILE = "new-dataset/training.csv"
TESTING_FILE = "new-dataset/testing.csv"
CLASSES = ("B", "M")
K = 5
P = 2


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


# PART B: KNN CLASSIFIER
# Assigned to: Calvin

# Calculate distance between two records using Minkowski distance
def calculate_distance(record_a, record_b, feature_names, p):
    total = 0.0

    for feature in feature_names:
        total += abs(record_a[feature] - record_b[feature]) ** p

    return total ** (1 / p)


# Find the k training records closest to the given record
def find_nearest_neighbors(training_rows, record, feature_names, k, p):
    distances = []

    # 1. calculate distance to each training record
    for train_record in training_rows:
        d = calculate_distance(record, train_record, feature_names, p)

        # 2. maintain sorted list of k smallest distances in descending order
        if len(distances) != k:
            distances.append((d, train_record["diagnosis"]))
            distances = sorted(distances, key=lambda item: item[0], reverse=True)
        elif d < distances[0][0]:
            distances.pop(0)
            distances.append((d, train_record["diagnosis"]))
            distances = sorted(distances, key=lambda item: item[0], reverse=True)

    # 3. return only the diagnosis labels of the k nearest neighbors
    return [item[1] for item in distances]


# Predict diagnosis using majority voting from nearest neighbor labels
def predict_one(neighbors):
    count_label = {}

    # 1. count occurrences of each diagnosis among neighbors
    for label in neighbors:
        if label in count_label:
            count_label[label] += 1
        else:
            count_label[label] = 1

    # 2. return the diagnosis with the highest vote count
    return max(count_label, key=count_label.get)


# PART C: EVALUATION
# Assigned to: Fae

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


# PART D: MAIN
# Assigned to: Calvin

def main():
    # code here
    pass


if __name__ == "__main__":
    main()
