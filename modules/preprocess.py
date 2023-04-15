import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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

def save_in_csv_file(df, path):
    """Save dataframe in csv file in the given path"""

    df.to_csv(path, index=False)

def delete_duplicates(df):
    """Delete duplicates from given Dataframe"""

    print("The shape of the data set before dropping duplicated:" + str(df.shape))

    df.drop_duplicates(inplace=True)

    print("The shape of the data set after dropping duplicated:" + str(df.shape))

def detect_outliers_IQR(df):
    """Returns the outliers of the given Dataframe"""

    # Calculate the Q1:
    Q1 = np.percentile(df, 25)

    # Calculate the Q3:
    Q3 = np.percentile(df, 75)

    #Calculate the IQR:
    IQR = Q3 - Q1

    # Upper bound
    upper = Q3 + 1.5 * IQR

    # Lower bound
    lower = Q1 - 1.5 * IQR

    # Outliers
    outliers = df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

    return outliers, upper, lower

def print_outliers_result(df, features_list):

    for feature in features_list:
        outliers, upper, lower = detect_outliers_IQR(df[feature])
        print(feature + 'Outliers Info: ')
        print("number of outliers: " + str(len(outliers)))
        print("max outlier value: " + str(outliers.max()))
        print("min of outliers: " + str(outliers.min()))
        print("Percentage of outliers: " + str(len(outliers) / len(df) * 100))
        print('----------------------------------------------')

def get_features_with_outliers(df):
    """Returns a list with the features of the given df that have outliers"""

    features_with_outliers = []

    features = get_features(df)
    features_list = features.columns

    for feature in features_list:
        outliers, upper, lower = detect_outliers_IQR(df[feature])
        if len(outliers) > 0:
            features_with_outliers.append(feature)

    print('Features with outliers: ', features_with_outliers)

    return features_with_outliers

def flooring_and_capping(df, features_list):
    """Applies flooring and capping technique to deal with outliers"""

    for feature in features_list:
        outliers, upper, lower = detect_outliers_IQR(df[feature])
        df[feature] = np.where(df[feature] > upper, upper, np.where(df[feature] < lower, lower, df[feature]))

def preprocess_room_occupancy_dataset(df):
    """Applies all data preprocessing steps to the given Dataframe"""

    # Step 1: Recode Room_Occupancy_Count column
    recode_dataset_output(df)

    # Step 2: Filter Date and Time columns
    remove_time_columns(df)

    # Step 3: Delete duplicate rows
    delete_duplicates(df)

    # Step 5: Treat outliers
    features_with_outliers = get_features_with_outliers(df)
    flooring_and_capping(df, features_with_outliers)

    return df
