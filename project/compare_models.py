"""
!!! DO NOT REMOVE !!!

model comparison and evaluation
assigned to fae and calvin

implementation
- run from-scratch implementations of knn, naive bayes, and perceptron
- evaluate all models on the identical 75/25 testing set
- collect accuracy, malignant recall, and confusion matrix counts directly from each model
- display a side-by-side terminal comparison table
- generate a bar graph figure for performance metrics
- generate a figure for the confusion matrices

evaluation
- use the existing 75/25 stratified training/testing split
- evaluate accuracy and malignant recall
- display confusion counts (TP, TN, FP, FN)
- treat malignant (M) as the positive class
- compare results across all three algorithms

notes
- do not modify or resplit the supplied datasets
- do not train or fit scaling parameters on testing data
- do not modify the original code in knn.py, naive_bayes.py, or perceptron.py
- may be modifed to be more interactive if connected to the GUI
"""

import os
import sys

# ensure local modules can be loaded regardless of execution directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# constants
TRAINING_FILE = os.path.join(PROJECT_DIR, "..", "new-dataset", "training.csv")
TESTING_FILE = os.path.join(PROJECT_DIR, "..", "new-dataset", "testing.csv")
CLASSES = ("B", "M")
FIGURES_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "figures"))


# PART A: EXECUTION
# Assigned to: ?

def run_knn():
    # code here
    pass


def run_naive_bayes():
    # code here
    pass


def run_perceptron():
    # code here
    pass



# PART B: FIGURE GENERATION - BAR GRAPH
# Assigned to: ?

def generate_bar_graph(models_data, output_dir=FIGURES_DIR):
    # code here
    pass


# PART C: FIGURE GENERATION - CONFUSION MATRIX
# Assigned to: ?

def generate_confusion_matrix_figures(models_data, output_dir=FIGURES_DIR):
    # code here
    pass


# PART D: TERMINAL PRINT and MAIN
# Assigned to: ?

def print_comparison_table(models_data):
    # code here
    pass

def main():
    # code here
    pass


if __name__ == "__main__":
    main()
