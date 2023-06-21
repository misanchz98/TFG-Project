from sklearn.metrics import precision_score, recall_score, f1_score

"""
This module provides all the functions you need to
calculate model's evaluation metrics for classification.
"""


def get_precision(y_test, y_pred):

    """
    Calculates precision metric using y_test and y_pred.

    Parameters
    ----------
    y_test : pandas.core.series.Series
        Output test data.

    y_pred : numpy.ndarray
        Predicted output.

    Returns
    -------
    numpy.float64
        Precision result.
    """

    return precision_score(y_test, y_pred)


def get_recall(y_test, y_pred):

    """
    Calculates recall metric using y_test and y_pred.

    Parameters
    ----------
    y_test : pandas.core.series.Series
        Output test data.

    y_pred : numpy.ndarray
        Predicted output.

    Returns
    -------
    numpy.float64
        Recall result.
    """

    return recall_score(y_test, y_pred)


def get_f_score(y_test, y_pred):

    """
    Calculates f score metric using y_test and y_pred.

    Parameters
    ----------
    y_test : pandas.core.series.Series
        Output test data.

    y_pred : numpy.ndarray
        Predicted output.

    Returns
    -------
    numpy.float64
        F score result.
    """

    return f1_score(y_test, y_pred)


def get_classification_metrics(y_pred, y_test):

    """
    Calculates the following classification metrics:
        - Precision.
        - Recall.
        - F score.

    Parameters
    ----------
    y_test : pandas.core.series.Series
        Output test data.

    y_pred : numpy.ndarray
        Predicted output.

    Returns
    -------
    precision : numpy.float64
        Precision result.

    recall : numpy.float64
        Recall result.

    f_score : numpy.float64
        F score result.
    """

    precision = get_precision(y_test, y_pred)
    recall = get_recall(y_test, y_pred)
    f_score = get_f_score(y_test, y_pred)

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f_score)

    return precision, recall, f_score


