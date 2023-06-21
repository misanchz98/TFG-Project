import pandas as pd
from modules import preprocess, tracking, evaluating


"""
This module has all the necessary functions to conduct the
benchmarking experiment.
"""


def create_benchmarking_structure():

    """
    Creates the DataFrame structure where we are going to store
    benchmarking results.

    Returns
    -------
    pandas.core.frame.DataFrame
        Benchmarking DataFrame.
    """

    df = pd.DataFrame({
        "Algoritmos": ['Logistic Regression', 'Random Forest', 'Support Vector Machines', 'Multilayer Perceptron'],
        "CodeCarbon (kWh)": [0, 0, 0, 0],
        "Eco2AI (kWh)": [0, 0, 0, 0],
        "Precision": [0, 0, 0, 0],
        "Recall": [0, 0, 0, 0],
        "F Score": [0, 0, 0, 0],
    })

    return df


def get_energy_consumed_codecarbon():

    """
    Gets the consumed energy results from codecarbon.csv.

    Returns
    -------
    String
        Consumed energy (kWh) results estimated by CodeCarbon.
    """

    df_codecarbon = preprocess.load_csv_data('codecarbon.csv')

    return df_codecarbon['energy_consumed']


def get_energy_consumed_eco2ai():

    """
    Gets the consumed energy results from eco2ai.csv.

    Returns
    -------
    string
        Consumed energy (kWh) results estimated by eco2AI.
    """

    df_eco2ai = preprocess.load_csv_data('eco2ai.csv')

    return df_eco2ai['power_consumption(kWh)']


def get_model_index(model):

    """
    Gets model's row index in benchmarking DataFrame giving its acronym:
        - LR: Logistic Regression.
        - RF: Random Forest.
        - SVM: Support Vector Machines.
        - MLP: Multilayer Perceptron.

    Parameters
    ----------
    model : string
        Model's acronym (LR, RF, SVM, MLP)

    Returns
    -------
    int
        Model's row index in benchmarking Dataframe
    """

    index = -1

    if model == 'LR':
        index = 0
    elif model == 'RF':
        index = 1
    elif model == 'SVM':
        index = 2
    elif model == 'MLP':
        index = 3

    return index


def get_model_energy_consumed(index):

    """
    Gets model's consumed energy estimated by CodeCarbon and eco2AI
    giving its row index in benchmarking DataFrame.

    Parameters
    ----------
    index : int
        Model's row index in benchmarking DataFrame.

    Returns
    -------
    model_energy_codecarbon : string
        Model's consumed energy estimated by CodeCarbon.

    model_energy_eco2ai : string
        Model's consumed energy estimated by eco2AI.
    """

    df_energy_codecarbon = get_energy_consumed_codecarbon()
    df_energy_eco2ai = get_energy_consumed_eco2ai()

    model_energy_codecarbon = df_energy_codecarbon.at[index]
    model_energy_eco2ai = df_energy_eco2ai.at[index]

    return model_energy_codecarbon, model_energy_eco2ai


def store_model_tracking_data(model, df_benchmarking):

    """
    Stores model's tracking data (consumed energy) in benchmarking DataFrame.

    Parameters
    ----------
    model : string
        Model's acronym (LR, RF, SVM, MLP).

    df_benchmarking : pandas.core.frame.DataFrame
        Benchmarking DataFrame.
    """

    model_index = get_model_index(model)

    model_energy_codecarbon, model_energy_eco2ai = get_model_energy_consumed(model_index)

    df_benchmarking.at[model_index, "CodeCarbon (kWh)"] = model_energy_codecarbon
    df_benchmarking.at[model_index, "Eco2AI (kWh)"] = model_energy_eco2ai


def store_model_metrics(model, y_pred, y_test, df_benchmarking):

    """
    First calculates classification metrics with y_pred and y_test and finally,
    stores the obtained results in benchmarking DataFrame.

    Parameters
    ----------
    model : string
        Model's acronym (LR, RF, SVM, MLP).

    y_pred : numpy.ndarray
        Predicted output.

    y_test : pandas.core.series.Series
        Output test data.

    df_benchmarking : pandas.core.frame.DataFrame
        Benchmarking DataFrame.
    """

    model_index = get_model_index(model)

    precision, recall, f_score = evaluating.get_classification_metrics(y_pred, y_test)

    df_benchmarking.at[model_index, "Precision"] = precision
    df_benchmarking.at[model_index, "Recall"] = recall
    df_benchmarking.at[model_index, "F Score"] = f_score


def create_benchmarking(X_train, y_train, X_test, y_test, cv=5):

    """
    Performs all the benchmarking process:
        - Step 1: Delete old result csv files (codecarbon.csv, eco2ai.csv).
        - Step 2: Create empty benchmarking DataFrame structure.
        - Step 3: Tracks training process (for each model).
        - Step 4: Store results in benchmarking DataFrame (for each model).

    Parameters
    ----------
    X_train : pandas.core.series.Series
        Features training data.

    y_train : pandas.core.series.Series
        Output training data.

    X_test : pandas.core.series.Series
        Features test data.

    y_test : pandas.core.series.Series
        Output test data.

    cv : int (default: 5)
        Number of splits in cross-validation.

    Returns
    -------
    pandas.core.frame.DataFrame
        Final benchmarking DataFrame with all results.
    """

    # Delete old result csv files
    preprocess.delete_csv_file('codecarbon.csv')
    preprocess.delete_csv_file('eco2ai.csv')

    # Create benchmarking Dataframe structure
    df_benchmarking = create_benchmarking_structure()

    # Store tracking and metrics results in benchmarking Dataframe
    models_list = ['LR', 'RF', 'SVM', 'MLP']

    for model in models_list:
        y_pred = tracking.track_model_training(model, X_train, y_train, X_test, cv)
        store_model_tracking_data(model, df_benchmarking)
        store_model_metrics(model, y_pred, y_test, df_benchmarking)

    return df_benchmarking


