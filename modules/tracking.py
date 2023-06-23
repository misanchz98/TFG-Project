from codecarbon import EmissionsTracker
from eco2ai import Tracker
from modules import tuning

"""
This module offers the basic functions to execute the 
tracking process with CodeCarbon and eco2ai for each 
model.
"""


def LR_tracking(X_train, y_train, X_test, cv):

    """
    Tracks Logistic Regression training process with CodeCarbon and
    eco2AI and returns its prediction.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
         Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    cv :  int
        Number of splits in cross-validation.

    Returns
    -------
    numpy.ndarray
        Predicted output.
    """

    grid_search = tuning.get_grid_search_LR(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker(
        project_name="Training LR",
        tracking_mode='process',
        output_file="codecarbon.csv",
        measure_power_secs=15
    )

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training LR",
        alpha_2_code="ES-MD",
        cpu_processes="current",
        file_name="eco2ai.csv",
        measure_period=15,
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

    """
    Tracks Random Forest training process with CodeCarbon and
    eco2AI and returns its prediction.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
         Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    cv :  int
        Number of splits in cross-validation.

    Returns
    -------
    numpy.ndarray
        Predicted output.
    """

    randomized_search = tuning.get_randomized_search_RF(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker(
        project_name="Training RF",
        output_file="codecarbon.csv",
        tracking_mode='process',
        measure_power_secs=15
    )

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training RF",
        alpha_2_code="ES-MD",
        cpu_processes='current',
        file_name="eco2ai.csv",
        measure_period=15,
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

    """
    Tracks Support Vector Machine training process with CodeCarbon and
    eco2AI and returns its prediction.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
         Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    cv :  int
        Number of splits in cross-validation.

    Returns
    -------
    numpy.ndarray
        Predicted output.
    """

    grid_search = tuning.get_grid_search_SVM(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker(
        project_name="Training SVM",
        output_file="codecarbon.csv",
        tracking_mode="process",
        measure_power_secs=15
    )

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training SVM",
        alpha_2_code="ES-MD",
        cpu_processes='current',
        file_name="eco2ai.csv",
        measure_period=15,
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

    """
    Tracks Multilayer Perceptron training process with CodeCarbon and
    eco2AI and returns its prediction.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
         Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    cv :  int
        Number of splits in cross-validation.

    Returns
    -------
    numpy.ndarray
        Predicted output.
    """

    randomized_search = tuning.get_randomized_search_MLP(cv)

    # Track with codecarbon and eco2AI
    tracker_codecarbon = EmissionsTracker(
        project_name="Training MLP",
        output_file="codecarbon.csv",
        tracking_mode="process",
        measure_power_secs=15
    )

    tracker_eco2AI = Tracker(
        project_name="TFG Project",
        experiment_description="Training MLP",
        alpha_2_code="ES-MD",
        cpu_processes='current',
        file_name="eco2ai.csv",
        measure_period=15,
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

    """
    Tracks the given model training process with CodeCarbon and
    eco2AI and returns its prediction.

    Parameters
    ----------
    model : string
        Model's acronym (LR, RF, SVM, MLP).

    X_train : pandas.core.frame.DataFrame
         Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    cv :  int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    numpy.ndarray
        Predicted output.
    """

    if model == 'LR':
        y_pred = LR_tracking(X_train, y_train, X_test, cv)
    elif model == 'RF':
        y_pred = RF_tracking(X_train, y_train, X_test, cv)
    elif model == 'SVM':
        y_pred = SVM_tracking(X_train, y_train, X_test, cv)
    elif model == 'MLP':
        y_pred = MLP_tracking(X_train, y_train, X_test, cv)

    return y_pred
