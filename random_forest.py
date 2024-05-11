# coding:utf-8
import numpy as np

from base import BaseEstimator
from base_tree import information_gain, mse_criterion
from tree import Tree
import SMOTE


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        """Base class for RandomForest.

        Parameters
        ----------
        n_estimators : int
            The number of decision tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        criterion : str
            The function to measure the quality of a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features
        self._train()

    def _train(self):
        for tree in self.trees:
            tree.train(
                self.X,
                self.y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

    def _predict(self, X=None):
        raise NotImplementedError()


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion="entropy", smote = None):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            criterion=criterion,
        )

        if criterion == "entropy":
            self.criterion = information_gain
        else:
            raise ValueError()

        self.smote = smote
        # Initialize empty trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(criterion=self.criterion))

    def _predict(self, X=None):
        y_shape = np.unique(self.y).shape[0]
        predictions = np.zeros((X.shape[0], y_shape))

        for i in range(X.shape[0]):
            row_pred = np.zeros(y_shape)
            for tree in self.trees:
                row_pred += tree.predict_row(X[i, :])

            row_pred /= self.n_estimators
            predictions[i, :] = row_pred
        return predictions

    def fit(self, X, y):
        if self.smote is not None:
            X, y = self.smote.fit_generate(X, y)
        
        super(RandomForestClassifier, self).fit(X, y)
