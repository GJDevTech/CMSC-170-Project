# CMSC 170 Lab Project

Binary classification of breast tumors as **Benign (B)** or **Malignant (M)** using machine learning algorithms implemented from scratch

## Members

1. Avanzado, Calvin Gennel M.
2. Colosaga, Gerizim Josh L.
3. Gabion, Jericho Russ A.
4. Ramos, Yumie Zoe P.

## Dataset Used

- **Source:** [Breast Cancer Dataset by M Yasser H (Kaggle)](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- **Records:** 569 with 30 numerical features
- **Split:** 75% training and 25% testing, using stratified random sampling

The raw dataset is stored at `new-dataset/breast-cancer.csv`. The supplied split files are:

- `new-dataset/training.csv` — 427 records (268 benign, 159 malignant)
- `new-dataset/testing.csv` — 142 records (89 benign, 53 malignant)

The split is made separately within each diagnosis class using `random.seed(67)`. The code uses `id` only as an identifier and excludes it and `diagnosis` from the numerical features.

## Algorithms

1. K-Nearest Neighbors (KNN)
2. Categorical Naïve Bayes with discretization and Laplace smoothing
3. Perceptron

**Evaluation:** Accuracy, malignant recall, and confusion matrix

All three classifiers are implemented from scratch. Malignant (`M`) is treated as the positive class for the confusion matrix and recall calculation.

### K-Nearest Neighbors

- `k = 5`
- Euclidean distance (`p = 2`)
- Min-max scaling, with minimum and maximum values calculated from the training set only

### Categorical Naïve Bayes

- Each continuous feature is discretized into `low`, `medium`, or `high`
- The 33rd and 67th percentile boundaries are calculated from the training set only
- Laplace smoothing uses `alpha = 1`
- Class scores are calculated using log probabilities

### Perceptron

- Single-layer perceptron with 30 numerical features
- Min-max scaling fitted on the training set only
- `M` is encoded as `+1` and `B` as `-1`
- Learning rate `r = 0.1`, initial bias `0.5`, and a maximum of 100 epochs

## Requirements

- Python 3.8 or newer
- `matplotlib` for the comparison chart and confusion-matrix figures

The classifiers themselves use only Python’s standard library. `scikit-learn`, `pandas`, and other high-level machine-learning libraries are not required.

## Project Structure

```text
CMSC-170-Project/
├── new-dataset/
│   ├── breast-cancer.csv
│   ├── training.csv
│   └── testing.csv
├── project/
│   ├── compare_models.py
│   ├── data_split.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── perceptron.py
└── figures/
```

## Setup

Open a terminal in the repository root, which is the folder containing this `README.md`.

Install the only required external package:

```bash
python -m pip install matplotlib
```

On macOS or Linux, use `python3` if `python` is not recognized:

```bash
python3 -m pip install matplotlib
```

## Running the Code

### Run all models and generate figures

From the repository root:

```bash
python project/compare_models.py
```

This command evaluates KNN, Naïve Bayes, and the perceptron on the same existing `training.csv` and `testing.csv` files. It prints the confusion counts, accuracy, and malignant recall, then creates or overwrites these files in `figures/`:

- `metrics_comparison.png`
- `confusion_matrix_knn_k5.png`
- `confusion_matrix_naive_bayes.png`
- `confusion_matrix_perceptron.png`

### Re-create the deterministic train/test split

The supplied `training.csv` and `testing.csv` files should be used as-is when reproducing the reported results. If they need to be regenerated from `new-dataset/breast-cancer.csv`, run this from the repository root:

```bash
python project/data_split.py
```

`data_split.py` uses seed `67`, performs a stratified 75/25 split, and overwrites `new-dataset/training.csv` and `new-dataset/testing.csv`. After regenerating the files, run `python project/compare_models.py` again.

### Run one classifier only

From the repository root:

```bash
python project/knn.py
python project/naive_bayes.py
python project/perceptron.py
```

These commands print the selected model’s metrics but do not create the comparison figures. The standalone scripts expect the `new-dataset/` paths relative to the repository root.

## Reproducing the Reported Results

Using the supplied split files and running `python project/compare_models.py` produces the following results. The percentages are calculated on the 142-record testing set.

| Model | TP | TN | FP | FN | Accuracy | Malignant Recall |
|---|---:|---:|---:|---:|---:|---:|
| KNN (`k=5`) | 52 | 88 | 1 | 1 | 98.59% | 98.11% |
| Naïve Bayes | 50 | 84 | 5 | 3 | 94.37% | 94.34% |
| Perceptron | 52 | 87 | 2 | 1 | 97.89% | 98.11% |

The confusion-matrix layout is:

|  | Predicted M | Predicted B |
|---|---:|---:|
| Actual M | TP | FN |
| Actual B | FP | TN |

Results are reproducible when the same dataset files, split seed, feature columns, and model settings are used. The models do not fit scaling or discretization parameters on the testing data.

This project is for educational purposes only. The dataset is credited to its original sources above.

