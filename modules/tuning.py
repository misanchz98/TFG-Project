from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import warnings


def get_model_hyperparameters(model):
    """Given the model name, returns its hyperparameters
    and their current value"""

    if model == "LR":
        estimator = LogisticRegression()
    elif model == "RF":
        estimator = RandomForestClassifier()
    elif model == "SVM":
        estimator = svm.SVC(kernel="rbf")
    elif model == "MLP":
        estimator = MLPClassifier()

    return estimator.get_params()


def get_estimator(model):
    """Given the model name, returns its estimator in scikit learn"""

    if model == "LR":
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', LogisticRegression(max_iter=500))])
    elif model == "RF":
        estimator = RandomForestClassifier()
    elif model == "SVM":
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', svm.SVC())])
    elif model == "MLP":
        estimator = Pipeline([('scaler', StandardScaler()), ('estimator', MLPClassifier(max_iter=500))])

    return estimator


def get_param_grid(model):
    """Given the model name, returns its parameters grid"""

    if model == "LR":
        param_grid = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'estimator__solver': ['lbfgs', 'liblinear']
        }
    elif model == "RF":
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9],
            'max_features': ['log2', 'sqrt', None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [25, 50, 100, 150]}
    elif model == "SVM":
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100, 1000],
            'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'estimator__kernel': ['rbf']}
    elif model == "MLP":
        param_grid = {
            'estimator__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'estimator__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'estimator__solver': ['sgd', 'adam', 'lbfgs'],
            'estimator__alpha': [0.0001, 0.05],
            'estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],
        }

    return param_grid


def tuning_with_grid(model, X_train, y_train, X_test, y_test, cv=3):
    """Tunes model with GridSearchCV"""

    estimator = get_estimator(model)
    param_grid = get_param_grid(model)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=2)
    grid_search.fit(X_train, y_train)

    print('Best estimator: ', grid_search.best_estimator_)
    print('Test set score: ', grid_search.score(X_test, y_test))
    print('Mean cross-validated score of the best_estimator: ', grid_search.best_score_)

    return grid_search.best_estimator_


def tuning_with_randomized(model, X_train, y_train, X_test, y_test, cv=3):
    """Tunes model with RandomizedSearchCV"""

    estimator = get_estimator(model)
    param_grid = get_param_grid(model)
    randomized_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=cv, verbose=2)
    randomized_search.fit(X_train, y_train)

    print('Best estimator: ', randomized_search.best_estimator_)
    print('Test set score: ', randomized_search.score(X_test, y_test))
    print('Mean cross-validated score of the best_estimator: ', randomized_search.best_score_)

    return randomized_search.best_estimator_


def get_best_estimator(model, X_train, y_train, cv=2):
    """Tune model with GridSearchCV and RandomizedSearchCV and
    returns the tuned model with the biggest accuracy"""

    gscv_best_estimator, gscv_best_score = tuning_with_grid(model, X_train, y_train, cv)
    rscv_best_estimator, rscv_best_score = tuning_with_randomized(model, X_train, y_train, cv)

    if gscv_best_score >= rscv_best_score:
        best_estimator = gscv_best_estimator
    else:
        best_estimator = rscv_best_estimator

    return best_estimator
