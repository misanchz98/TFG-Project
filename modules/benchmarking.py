import pandas as pd
from modules import preprocess

def create_benchmarking_df():
    """Creates basic benchmarking Dataframe"""

    benchmarking = {
        "Herramientas": ['codecarbon', 'eco2ai'],
        "Logistic Regression(kWh)": [0, 0],
        "Random Forest(kWh)": [0, 0],
        "Support Vector Machines(kWh)": [0, 0],
        "Multilayer Perceptron(kWh)": [0, 0],
        "Eficiencia energetica": [0, 0],
    }

    df = pd.DataFrame(benchmarking)

    return df

def get_column_by_index(index):
    """Returns benchmarking's column name given index result"""

    if index == 0:
        column = 'Logistic Regression(kWh)'
    elif index == 1:
        column = 'Random Forest(kWh)'
    elif index == 2:
        column = 'Support Vector Machines(kWh)'
    elif index == 3:
        column = 'Multilayer Perceptron(kWh)'

    return column

def create_benchmarking_csv():
    # Create codecarbon and eco2ai Dataframes
    df_codecarbon = preprocess.load_csv_data('emissions.csv')
    df_eco2ai = preprocess.load_csv_data('emission.csv')

    # Create Dataframe for benchmarking
    df_benchmarking = create_benchmarking_df()

    # Take results from codecarbon and eco2ai Dataframes
    energy_codecarbon = df_codecarbon['energy_consumed']
    energy_eco2ai = df_eco2ai['power_consumption(kWh)']

    # Save results in benchmarking Dataframe

    # Codecarbon
    for index, result in enumerate(energy_codecarbon):
        column = get_column_by_index(index)
        df_benchmarking.at[0, column] = result

    # Eco2AI
    for index, result in enumerate(energy_eco2ai):
        column = get_column_by_index(index)
        df_benchmarking.at[1, column] = result

    # Save benchmarking Dataframe in csv file
    preprocess.save_in_csv_file(df_benchmarking, 'benchmarking.csv')
