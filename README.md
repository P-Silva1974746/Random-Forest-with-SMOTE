# Random Forest with SMOTE

A from-scratch Random Forest classifier where **SMOTE** (Synthetic Minority Over-sampling Technique) is applied *per tree*, so each tree in the ensemble can be trained on a differently balanced version of the data. The goal is to study whether this improves performance on imbalanced classification datasets from [OpenML](https://www.openml.org/).

Project for the Machine Learning (AC) course, 2nd year.

## How it works

`RandomForestClassifier` takes a `smote` instance and a `smote_type`. Before each tree is trained, SMOTE generates synthetic minority-class samples. The amount generated is a percentage of what it would take to fully balance each minority class against the majority class:

| `smote_type` | Oversampling percentage per tree |
|---|---|
| `None` | No SMOTE (plain Random Forest) |
| `"binary"` | Alternates between 0% and 100% (trees 1, 3, 5… use the original data; trees 2, 4, 6… use fully balanced data) |
| `"gradient"` | `[0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1]`, a gradual increase across trees (needs exactly `n_estimators=10`) |

## Repository structure

| File | Description |
|---|---|
| `random_forest.py` | `RandomForest` / `RandomForestClassifier`, with per-tree SMOTE |
| `tree.py` | Recursive decision tree implementation |
| `base.py` | `BaseEstimator`, which handles input validation and the fit/predict interface |
| `base_tree.py` | Split criteria (entropy / information gain, MSE) and dataset splitting helpers |
| `SMOTE.py` | SMOTE implementation in PyTorch (k-nearest neighbours, Euclidean distance) |
| `metrics.py` | Evaluation metrics (accuracy, MSE, RMSE, MAE, log loss, …) |
| `test.py` | Minimal example: trains on OpenML dataset 11 with binary SMOTE |
| `test.ipynb` | Experiments across all the evaluated datasets |
| `graphs.ipynb` | Plots comparing no SMOTE, binary SMOTE and gradient SMOTE |
| `graphs_info.txt` | Results table per dataset (accuracy, AUC, balanced accuracy, MSE, RMSE, MAE) |
| `datasets_notebook_printed_format/` | HTML exports of the notebook results per OpenML dataset |
| `Machine_Learning(1).pptx` | Project presentation |
| `weather.csv` | Small toy dataset |

## Installation

```bash
git clone git@github.com:P-Silva1974746/Random-Forest-with-SMOTE.git
cd Random-Forest-with-SMOTE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project was developed with Python 3.10. `requirements.txt` lists only the main libraries, without pinned versions.

## Usage

Run the example script (it downloads OpenML dataset 11 the first time):

```bash
python test.py
```

Or use the classifier directly:

```python
import openml
from sklearn.model_selection import train_test_split
from random_forest import RandomForestClassifier
from SMOTE import smote

dataset = openml.datasets.get_dataset(11)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
y = y.astype("category").cat.codes          # encode labels as 0..n-1

X_train, X_test, y_train, y_test = train_test_split(X, y)

smote_i = smote(distance="euclidian", dims=X.shape[1], k=5)   # dims = number of features
model = RandomForestClassifier(n_estimators=10, max_depth=15, smote=smote_i, smote_type="gradient")
model.fit(X_train, y_train)
probas = model.predict(X_test)              # class probabilities, shape (n_samples, n_classes)
```

To reproduce the experiments and plots, open the notebooks:

```bash
jupyter notebook test.ipynb
jupyter notebook graphs.ipynb
```

## Results

The models were evaluated on 10 OpenML datasets (IDs 1462, 44, 40982, 1475, 1497, 11, 1489, 1063, 1067, 1049) with varying levels of class imbalance. The full table is in `graphs_info.txt`.

On the more imbalanced datasets (e.g. 1063, 1067, 1049), the SMOTE variants clearly improve **balanced accuracy** and **AUC** over the plain Random Forest. On nearly balanced datasets the three approaches perform about the same.

## Notes

- Labels must be integer-encoded (`0..n_classes-1`), because SMOTE uses them as indices.
- Features must be numeric.
- The decision tree and base estimator code is adapted from [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms). The SMOTE implementation is based on a PyTorch version by nuvilabs.
