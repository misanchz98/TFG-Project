from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

"""
This module presents the fundamental functions to 
execute the model's tuning.
"""


def get_grid_search_LR(cv):

    """
    Creates Grid search for Logistic Regression
    hyperparamenters tuning.

    Parameters
    ----------
    cv :  int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    sklearn.model_selection._search.GridSearchCV
        Grid search for Logistic Regression.
    """

    pipeline = Pipeline([('scaler', RobustScaler()), ('estimator', LogisticRegression())])

    param_grid = {
        'estimator__penalty': ['l1', 'l2'],
        'estimator__C': np.logspace(-4, 4, 20),
        'estimator__solver': ['liblinear'],
        'estimator__max_iter': [400, 500, 600]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=1)

    return grid_search


def get_grid_search_SVM(cv):

    """
    Creates Grid search for Support Vector Machine
    hyperparamenters tuning.

    Parameters
    ----------
    cv :  int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    sklearn.model_selection._search.GridSearchCV
        Grid search for Support Vector Machine.
    """

    estimator = svm.SVC()

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=1)

    return grid_search


def get_randomized_search_RF(cv):

    """
    Creates Randomized search for Random Forest
    hyperparamenters tuning.

    Parameters
    ----------
    cv :  int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    sklearn.model_selection._search.RandomizedSearchCV
        Randomized search for Random Forest.
    """

    estimator = RandomForestClassifier()

    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    randomized_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=cv, verbose=1)

    return randomized_search


def get_randomized_search_MLP(cv):

    """
    Creates Randomized search for Multilayer Perceptron
    hyperparamenters tuning.

    Parameters
    ----------
    cv :  int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    sklearn.model_selection._search.RandomizedSearchCV
        Randomized search for Multilayer Perceptron.
    """

    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', MLPClassifier())])

    param_grid = {
        'estimator__hidden_layer_sizes': [(100, 100, 100), (100, 150, 100)],
        'estimator__activation': ['identity'],
        'estimator__solver': ['adam'],
        'estimator__alpha': [0.03, 0.031, 0.032, 0.033, 0.034, 0.035],
        'estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'estimator__max_iter': [625, 650]
    }

    randomized_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=cv, verbose=1)

    return randomized_search
