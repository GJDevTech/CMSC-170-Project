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

import knn
import naive_bayes
import perceptron

# constants
TRAINING_FILE = os.path.join(PROJECT_DIR, "..", "new-dataset", "training.csv")
TESTING_FILE = os.path.join(PROJECT_DIR, "..", "new-dataset", "testing.csv")
CLASSES = ("B", "M")
FIGURES_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "figures"))


# PART A: EXECUTION
# Assigned to: Fae

def run_knn():
    # run the full knn pipeline from knn.py and return its results dictionary
    training_rows, features = knn.load_dataset(TRAINING_FILE)
    testing_rows, test_features = knn.load_dataset(TESTING_FILE)

    # confirm feature columns match
    if features != test_features:
        raise ValueError("Training and testing features do not match.")

    # min-max values come from training data only, then scale both sets
    min_max = knn.calculate_min_max(training_rows, features)
    scaled_training = knn.scale_dataset(training_rows, features, min_max)
    scaled_testing = knn.scale_dataset(testing_rows, features, min_max)

    # predict each testing record from its k nearest neighbors
    predictions = []
    for record in scaled_testing:
        neighbors = knn.find_nearest_neighbors(scaled_training, record, features, knn.K, knn.P)
        prediction = knn.predict_one(neighbors)
        predictions.append(prediction)

    # evaluate against the actual diagnoses
    results = knn.evaluate_predictions(scaled_testing, predictions)
    return results

def run_naive_bayes():
    # run the full naive bayes pipeline from naive_bayes.py and return its results dictionary
    training_rows, features = naive_bayes.load_dataset(TRAINING_FILE)
    testing_rows, test_features = naive_bayes.load_dataset(TESTING_FILE)

    # confirm feature columns match
    if features != test_features:
        raise ValueError("Training and testing features do not match.")

    # boundaries come from training data only, then categorize both sets
    boundaries = naive_bayes.calculate_boundaries(training_rows, features)
    categorized_training = naive_bayes.categorize_dataset(training_rows, features, boundaries)
    categorized_testing = naive_bayes.categorize_dataset(testing_rows, features, boundaries)

    # priors and likelihoods come from the categorized training rows
    priors, class_counts = naive_bayes.calculate_priors(categorized_training)
    likelihoods = naive_bayes.calculate_likelihoods(categorized_training, features, class_counts)

    # predict each testing record
    predictions = []
    for record in categorized_testing:
        prediction = naive_bayes.predict_one(record, features, priors, likelihoods)
        predictions.append(prediction)

    # evaluate against the actual diagnoses
    results = naive_bayes.evaluate_predictions(categorized_testing, predictions)
    return results


def run_perceptron():
    # run the full perceptron pipeline from perceptron.py and return its results dictionary
    training_rows, features = perceptron.load_dataset(TRAINING_FILE)
    testing_rows, test_features = perceptron.load_dataset(TESTING_FILE)

    # confirm feature columns match
    if features != test_features:
        raise ValueError("Training and testing features do not match.")

    # min-max values come from training data only, then scale both sets
    min_max = perceptron.calculate_min_max(training_rows, features)
    scaled_training = perceptron.scale_dataset(training_rows, features, min_max)
    scaled_testing = perceptron.scale_dataset(testing_rows, features, min_max)

    # train on the scaled training rows only
    weights, bias, epochs_run, converged = perceptron.train_perceptron(scaled_training, features, perceptron.LEARNING_RATE, perceptron.MAX_EPOCHS, perceptron.INITIAL_BIAS)

    # predict the testing rows with the final weights and bias
    predictions = perceptron.predict_dataset(scaled_testing, features, weights, bias)

    # evaluate against the actual diagnoses
    results = perceptron.evaluate_predictions(scaled_testing, predictions)
    return results


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
# Assigned to: Fae

def generate_confusion_matrix_figures(models_data, output_dir=FIGURES_DIR):
    # save one confusion matrix image per model (malignant = positive class)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed.")
        return

    # make sure the output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # rows are the actual class (M, B), columns are the predicted class (M, B)
    cell_names = [["TP", "FN"], ["FP", "TN"]]

    for model in models_data:
        TP = models_data[model]["TP"]
        TN = models_data[model]["TN"]
        FP = models_data[model]["FP"]
        FN = models_data[model]["FN"]

        matrix = [[TP, FN], [FP, TN]]
        max_value = max(TP, TN, FP, FN)

        # draw the matrix, a darker color means a higher count
        plt.figure(figsize=(5.5, 5))
        plt.imshow(matrix, cmap="Blues", vmin=0, vmax=max_value)

        # write the cell name and count inside each cell
        for row in range(2):
            for col in range(2):
                value = matrix[row][col]

                # use white text on dark cells so it stays readable
                if value > max_value / 2:
                    text_color = "white"
                else:
                    text_color = "black"

                plt.text(col, row, f"{cell_names[row][col]}\n{value}", ha="center", va="center", color=text_color, fontsize=18, fontweight="bold")

        # label the axes and add the title
        plt.xticks([0, 1], ["Predicted M", "Predicted B"], fontsize=10, fontweight="bold")
        plt.yticks([0, 1], ["Actual M", "Actual B"], fontsize=10, fontweight="bold")
        plt.title(f"Confusion Matrix: {model}", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        # build the file name from the model name (KNN (k=5) -> knn_k5)
        name = model.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        file_path = os.path.join(output_dir, "confusion_matrix_" + name + ".png")

        # save the figure
        with open(file_path, "wb") as f:
            plt.savefig(f, format="png", dpi=300)
        plt.close()

        print(f"Successfully saved {file_path}")


# PART D: TERMINAL PRINT and MAIN
# Assigned to: Calvin

def print_comparison_table(models_data):
    print("\n========== MODEL COMPARISON RESULTS ==========")

    for model, results in models_data.items():
        print(f"\n--- {model} ---")
        print(f"True Positive (TP):  {results['TP']}")
        print(f"True Negative (TN):  {results['TN']}")
        print(f"False Positive (FP): {results['FP']}")
        print(f"False Negative (FN): {results['FN']}")
        print(f"Accuracy:          {results['accuracy'] * 100:.2f}%")
        print(f"Malignant Recall:  {results['malignant_recall'] * 100:.2f}%")

    print("\n==============================================")


def main():
    print("Running model evaluations...\n")

    # 1. execute each model pipeline
    knn_results = run_knn()
    nb_results = run_naive_bayes()
    perceptron_results = run_perceptron()

    # 2. collect all evaluation results into a shared dictionary
    models_data = {
        "KNN (k=5)": knn_results,
        "Naive Bayes": nb_results,
        "Perceptron": perceptron_results,
    }

    # 3. print comparison table in terminal
    print_comparison_table(models_data)

    # 4. generate comparison bar graph
    print("\nGenerating performance metrics bar graph...")
    generate_bar_graph(models_data, output_dir=FIGURES_DIR)

    # 5. generate confusion matrix figures for each model
    print("Generating confusion matrix figures...")
    generate_confusion_matrix_figures(models_data, output_dir=FIGURES_DIR)

    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    main()
