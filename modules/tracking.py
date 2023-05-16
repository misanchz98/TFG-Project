from codecarbon import EmissionsTracker
from eco2ai import Tracker
from modules import tuning
import warnings


def LR_tracking(X_train, y_train, X_test, cv):
    """Tracks Logistic Regression training process with CodeCarbon and
    Eco2AI and returns the prediction"""

    #warnings.filterwarnings('ignore')
    grid_search = tuning.get_grid_search_LR(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker()

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training LR",
        alpha_2_code="ES-MD",
        ignore_warnings=True
    )

    tracker_codecarbon.start()
    tracker_eco2AI.start()

    grid_search.fit(X_train, y_train)

    tracker_codecarbon.stop()
    tracker_eco2AI.stop()

    print('Best estimator LR: ', grid_search.best_estimator_)
    y_pred = grid_search.predict(X_test)

    return y_pred


def RF_tracking(X_train, y_train, X_test, cv):
    """Tracks Random Forest training process with CodeCarbon and
    Eco2AI and returns the prediction"""

    randomized_search = tuning.get_randomized_search_RF(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker()

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training RF",
        alpha_2_code="ES-MD",
        ignore_warnings=True
    )

    tracker_codecarbon.start()
    tracker_eco2AI.start()

    randomized_search.fit(X_train, y_train)

    tracker_codecarbon.stop()
    tracker_eco2AI.stop()

    print('Best estimator RF: ', randomized_search.best_estimator_)
    y_pred = randomized_search.predict(X_test)

    return y_pred


def SVM_tracking(X_train, y_train, X_test, cv):
    """Tracks Support Vector Machine training process with CodeCarbon and
    Eco2AI and returns the prediction"""

    grid_search = tuning.get_grid_search_SVM(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker()

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training SVM",
        alpha_2_code="ES-MD",
        ignore_warnings=True
    )

    tracker_codecarbon.start()
    tracker_eco2AI.start()

    grid_search.fit(X_train, y_train)

    tracker_codecarbon.stop()
    tracker_eco2AI.stop()

    print('Best estimator SVM: ', grid_search.best_estimator_)

    y_pred = grid_search.predict(X_test)

    return y_pred


def MLP_tracking(X_train, y_train, X_test, cv):
    """Tracks Multilayer Perceptron training process with CodeCarbon and
    Eco2AI and returns the prediction"""

    randomized_search = tuning.get_randomized_search_MLP(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker()

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training MLP",
        alpha_2_code="ES-MD",
        ignore_warnings=True
    )

    tracker_codecarbon.start()
    tracker_eco2AI.start()

    randomized_search.fit(X_train, y_train)

    tracker_codecarbon.stop()
    tracker_eco2AI.stop()

    print('Best estimator MLP: ', randomized_search.best_estimator_)
    y_pred = randomized_search.predict(X_test)

    return y_pred


def track_model_training(model, X_train, y_train, X_test, cv=5):
    if model == 'LR':
        y_pred = LR_tracking(X_train, y_train, X_test, cv)
    elif model == 'RF':
        y_pred = RF_tracking(X_train, y_train, X_test, cv)
    elif model == 'SVM':
        y_pred = SVM_tracking(X_train, y_train, X_test, cv)
    elif model == 'MLP':
        y_pred = MLP_tracking(X_train, y_train, X_test, cv)

    return y_pred
