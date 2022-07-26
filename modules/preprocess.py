import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load data as DataFrame
def load_csv_data(path):
    """Load and return data from csv file as DataFrame"""

    df = pd.read_csv(path)

    return df

# Step 2: Recode RoomOccupancyCount column
def recode_dataset_output(df):
    """Recode dataset's output, if Room_Occupancy_Count > 0, we change its value into 1"""

    df["Room_Occupancy_Count"] = np.where(df["Room_Occupancy_Count"] > 0, 1, 0)


# Step 3: Remove of time columns (Time and Date)
def remove_time_columns(df):
    """Remove Time and Date columns"""

    df.drop(['Time'], axis=1, inplace=True)
    df.drop(['Date'], axis=1, inplace=True)


# Step 4: Split dataset into train and test
def split_dataset(df, train_size=None):
    """Split dataset into train and test"""

    train_df, test_df = train_test_split(df, train_size=train_size)

    return train_df, test_df

# Step 5: Split train dataset into features (X) and output (y)
def get_features(df):
    """Transform the dataset into another dataset but only with feature columns"""

    df_copy = df.copy()
    df_copy.drop(['Room_Occupancy_Count'], axis=1, inplace=True)

    return df_copy

def get_output(df):
    """Get the output column from dataset"""

    output = df["Room_Occupancy_Count"]

    return output
