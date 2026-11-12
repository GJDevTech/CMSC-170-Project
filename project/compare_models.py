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
# Assigned to: Calvin

# Generate a bar graph comparing accuracy and malignant recall
def generate_bar_graph(models_data, output_dir=FIGURES_DIR):

    # 1. check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed.")
        return

    # 2. ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. define metrics and extract values
    models = list(models_data.keys())
    metrics = ("Accuracy", "Malignant Recall")
    metric_keys = ("accuracy", "malignant_recall")

    x = range(len(metrics))
    width = 0.25
    colors = ("#FF0000", "#00FF00", "#0000FF")

    # 4. create bar chart
    plt.figure(figsize=(8, 5.5)) 

    for i, model in enumerate(models):
        # convert accuracy and recall decimals into percentages (e.g. 0.98 -> 98.0)
        values = [models_data[model][key] * 100 for key in metric_keys]

        # shift bars so the 3 models stand side-by-side instead of overlapping:
        # i=0 shifts left (-width), i=1 stays center (0), i=2 shifts right (+width)
        offset = (i - 1) * width

        # plot the bars at their shifted x-positions
        bars = plt.bar([pos + offset for pos in x], values, width=width, label=model, color=colors[i % len(colors)], edgecolor="black")

        # add percentage label on top of each bar
        for bar in bars:
            height = bar.get_height()
            text_x = bar.get_x() + bar.get_width() / 2.0
            text_y = height + 0.3
            plt.text(text_x, text_y, f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", clip_on=False)

    # 5. configure plot labels and style
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Performance Comparison Across All Three Algorithms", fontsize=13, fontweight="bold", pad=14)
    plt.xticks(x, metrics, fontsize=10, fontweight="bold")
    plt.ylim(80, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 6. save figure
    chart_path = os.path.join(output_dir, "metrics_comparison.png")
    with open(chart_path, "wb") as file:
        plt.savefig(file, format="png", dpi=300)
    plt.close()

    print(f"Successfully saved {chart_path}")


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
