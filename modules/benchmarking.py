import pandas as pd

def create_benchmarking_df():
    """Creates basic benchmarking Dataframe"""

    benchmarking = {
        "Herramientas": ['codecarbon', 'eco2ai'],
        "Logistic Regression": [0, 0],
        "Random Forest": [0, 0],
        "Support Vector Machines": [0, 0],
        "Multilayer Perceptron": [0, 0],
        "Eficiencia energetica": [0, 0],
    }

    df = pd.DataFrame(benchmarking)

    return df