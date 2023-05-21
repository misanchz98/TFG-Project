import pandas as pd
from modules import preprocess, tracking, evaluating


def create_benchmarking_structure():
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
    df_codecarbon = preprocess.load_csv_data('codecarbon.csv')

    return df_codecarbon['energy_consumed']


def get_energy_consumed_eco2ai():
    df_eco2ai = preprocess.load_csv_data('eco2ai.csv')

    return df_eco2ai['power_consumption(kWh)']


def get_model_energy_consumed(index):
    df_energy_codecarbon = get_energy_consumed_codecarbon()
    df_energy_eco2ai = get_energy_consumed_eco2ai()

    model_energy_codecarbon = df_energy_codecarbon.at[index]
    model_energy_eco2ai = df_energy_eco2ai.at[index]

    return model_energy_codecarbon, model_energy_eco2ai


def get_model_index(model):
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


def store_model_tracking_data(model, df_benchmarking):
    model_index = get_model_index(model)

    model_energy_codecarbon, model_energy_eco2ai = get_model_energy_consumed(model_index)

    df_benchmarking.at[model_index, "CodeCarbon (kWh)"] = model_energy_codecarbon
    df_benchmarking.at[model_index, "Eco2AI (kWh)"] = model_energy_eco2ai


def store_model_metrics(model, y_pred, y_test, df_benchmarking):
    model_index = get_model_index(model)

    precision, recall, f_score = evaluating.get_classification_metrics(y_pred, y_test)

    df_benchmarking.at[model_index, "Precision"] = precision
    df_benchmarking.at[model_index, "Recall"] = recall
    df_benchmarking.at[model_index, "F Score"] = f_score


def create_benchmarking(X_train, y_train, X_test, y_test, cv=5):
    # Delete old result csv files
    preprocess.delete_csv_file('codecarbon.csv')
    preprocess.delete_csv_file('eco2ai.csv')

    # Create benchmarking Dataframe structure
    df_benchmarking = create_benchmarking_structure()

    models_list = ['LR', 'RF', 'SVM', 'MLP']

    for model in models_list:
        y_pred = tracking.track_model_training(model, X_train, y_train, X_test, cv)
        store_model_tracking_data(model, df_benchmarking)
        store_model_metrics(model, y_pred, y_test, df_benchmarking)

    return df_benchmarking


