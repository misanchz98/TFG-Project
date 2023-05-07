import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def load_csv_data(path):
    """Load and return data from csv file as DataFrame"""

    df = pd.read_csv(path)

    return df


# Step 2: Recode RoomOccupancyCount column
def recode_dataset_output(df):
    """Recode dataset's output, if Room_Occupancy_Count > 0, we change its value into 1"""

    df["Room_Occupancy_Count"] = np.where(df["Room_Occupancy_Count"] > 0, 1, 0)


def remove_time_columns(df):
    """Remove Time and Date columns"""

    df.drop(['Time'], axis=1, inplace=True)
    df.drop(['Date'], axis=1, inplace=True)


def split_dataset(X, y, test_size=0.3):
    """Splits X (features) and y (output) into train and test"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def get_features(df):
    """Transform the dataset into another dataset but only with feature columns"""

    df_copy = df.copy()
    df_copy.drop(['Room_Occupancy_Count'], axis=1, inplace=True)

    return df_copy


def get_output(df):
    """Get the output column from dataset"""

    output = df["Room_Occupancy_Count"]

    return output


def get_number_duplicated_rows(df):
    """Prints the number of duplicated rows in the given Dataframe"""

    num_dups = df.duplicated().sum()

    print(f"Number of duplicated rows: {num_dups}")


def delete_duplicates(df):
    """Delete duplicates from given Dataframe"""

    print("The shape of the data set before dropping duplicated:" + str(df.shape))

    df.drop_duplicates(inplace=True)

    print("The shape of the data set after dropping duplicated:" + str(df.shape))


def delete_csv_file(file):
    if os.path.exists(file):
        os.remove(file)


def save_in_csv_file(df, path):
    """Save dataframe in csv file in the given path"""

    df.to_csv(path, index=False)
