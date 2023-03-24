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
    """Given the model name, return its hyperparameters
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


def create_pipeline(model):
    if model == "LR":
        estimator = LogisticRegression(max_iter=500)
    elif model == "RF":
        estimator = RandomForestClassifier()
    elif model == "SVM":
        estimator = svm.SVC()
    elif model == "MLP":
        estimator = MLPClassifier(max_iter=500)

    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', estimator)])

    return pipeline


def get_param_grid(model):
    if model == "LR":
        param_grid = {
            'estimator__penalty': ['l1', 'l2', 'none'],
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'estimator__solver': ['lbfgs', 'liblinear']}
    elif model == "RF":
        param_grid = {
            'estimator__bootstrap': [True, False],
            'estimator__max_depth': [3, 6, 9],
            'estimator__max_leaf_nodes': [3, 6, 9],
            'estimator__max_features': ['log2', 'sqrt', None],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__n_estimators': [25, 50, 100, 150]}

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


def tuning_with_grid(model, X_train, y_train, cv=10):
    warnings.filterwarnings('ignore')
    pipeline = create_pipeline(model)
    param_grid = get_param_grid(model)
    gscv = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=2)
    gscv.fit(X_train, y_train)

    return gscv.best_estimator_, gscv.best_score_


def tuning_with_randomized(model, X_train, y_train, cv=5):
    pipeline = create_pipeline(model)
    param_grid = get_param_grid(model)
    rscv = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=cv, verbose=2)
    rscv.fit(X_train, y_train)

    return rscv.best_estimator_, rscv.best_score_


def get_best_estimator(model, X_train, y_train, cv=5):
    gscv_best_estimator, gscv_best_score = tuning_with_grid(model, X_train, y_train, cv)
    print("Best estimator Grid: ", gscv_best_estimator)
    print("Accuracy Grid: ", gscv_best_score)

    rscv_best_estimator, rscv_best_score = tuning_with_randomized(model, X_train, y_train, cv)
    print("Best estimator Randomized: ", rscv_best_estimator)
    print("Accuracy Randomized: ", rscv_best_score)

    if gscv_best_score >= rscv_best_score:
        best_estimator = gscv_best_estimator
    else:
        best_estimator = rscv_best_estimator

    print("FINAL BEST ESTIMATOR: ", best_estimator)

    return best_estimator
