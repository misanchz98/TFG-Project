import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

"""
This module supplies the vital functions to perform 
data preprocessing and csv files treatment.
"""


def load_csv_data(path):

    """
    Loads data from csv file as DataFrame.

    Parameters
    ----------
    path : string
       Csv file path.

    Returns
    -------
    pandas.core.frame.DataFrame
        DataFrame with csv file data.
    """

    df = pd.read_csv(path)

    return df


def delete_csv_file(file):

    """
    Deletes the given csv file.

    Parameters
    ----------
    file : string
       Csv file.
    """

    if os.path.exists(file):
        os.remove(file)


def save_in_csv_file(df, path):

    """
    Saves DataFrame in csv file in the given path.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
       DataFrame we want to save in csv file.

    path : string
        Path where we want to store the csv file.
    """

    df.to_csv(path, index=False)


def recode_dataset_output(df):

    """
    Recodes dataset's output, if Room_Occupancy_Count > 0, we change its value into 1.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
       Dataset DataFrame.
    """

    df["Room_Occupancy_Count"] = np.where(df["Room_Occupancy_Count"] > 0, 1, 0)


def remove_time_columns(df):

    """
    Removes Time and Date columns.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
       Dataset DataFrame.
    """

    df.drop(['Time'], axis=1, inplace=True)
    df.drop(['Date'], axis=1, inplace=True)


def get_features(df):

    """
    Gets dataset's features columns.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataset DataFrame.

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataset's features.
    """

    df_copy = df.copy()
    df_copy.drop(['Room_Occupancy_Count'], axis=1, inplace=True)

    return df_copy


def get_output(df):

    """
    Gets dataset's output column.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataset DataFrame.

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataset's features.
    """

    output = df["Room_Occupancy_Count"]

    return output


def split_dataset(X, y, test_size=0.3):

    """
    Splits dataset's features (X) and output (y) into train and test.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Dataset's features.

    y: pandas.core.series.Series
        Dataset's output.

    test_size : float (default: 0.3)
        Size of test set.

    Returns
    -------
    X_train : pandas.core.frame.DataFrame
        Features training data.

    X_test : pandas.core.frame.DataFrame
        Features test data.

    y_train : pandas.core.series.Series
        Output training data.

    y_test : pandas.core.series.Series
        Output test data.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

