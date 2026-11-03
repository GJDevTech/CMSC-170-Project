import csv
import random

# load dataset
with open("new-dataset/breast-cancer.csv", "r", newline="") as file:
    reader = csv.DictReader(file)

    # match col and rows
    columns = reader.fieldnames
    rows = list(reader)

# random seed so results would be consistent
random.seed(67)

# init. containers
training = []
testing = []

# split each diagnosis class into 75% training and 25% testing
for label in ("B", "M"):
    group = [row for row in rows if row["diagnosis"] == label]

    random.shuffle(group)  # shuffle so that split would be random
    split_index = round(len(group) * 0.75)  # calculate where to perform split

    # perform split
    training.extend(group[:split_index])
    testing.extend(group[split_index:])

# shuffle both sets to mix the diagnosis classes
random.shuffle(training)
random.shuffle(testing)

# save training dataset
with open("new-dataset/training.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(training)
    print("Successfully created new-dataset/training.csv")

# save testing dataset
with open("new-dataset/testing.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(testing)
    print("Successfully created new-dataset/testing.csv")

print("Training:", len(training))
print("Testing:", len(testing))