import matplotlib.pyplot as plt
import numpy as np

"""
This modules gives the essential tools to visualize
benchmarking results.
"""


def plot_models_energy_consumed(df_benchmarking):

    """
    Plots benchmarking consumed energy results for each model.

    Parameters
    ----------
    df_benchmarking : pandas.core.frame.DataFrame
        Benchmarking DataFrame.
    """

    models = ['LR', 'RF', 'SVM', 'MLP']
    codecarbon_energy = df_benchmarking['CodeCarbon (kWh)']
    eco2ai_energy = df_benchmarking['Eco2AI (kWh)']

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))

    fig.subplots_adjust(wspace=0.5)

    ax[0].bar(models, codecarbon_energy, color='green')
    ax[0].set_xlabel('Modelos')
    ax[0].set_ylabel('Energía consumida (kWh)')
    ax[0].set_title('CodeCarbon')

    ax[1].bar(models, eco2ai_energy, color='blue')
    ax[1].set_xlabel('Modelos')
    ax[1].set_ylabel('Energía consumida (kWh)')
    ax[1].set_title('Eco2AI')

    plt.show()


def plot_models_evaluation_metrics(df_benchmarking):

    """
    Plots benchmarking classification metrics results for each model.

    Parameters
    ----------
    df_benchmarking : pandas.core.frame.DataFrame
        Benchmarking DataFrame.
    """

    models = ['LR', 'RF', 'SVM', 'MLP']

    precision = df_benchmarking['Precision']
    recall = df_benchmarking['Recall']
    f_score = df_benchmarking['F Score']

    n = len(models)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots()

    ax.bar(x - width, precision, width=width, label='Precision')
    ax.bar(x, recall, width=width, label='Recall')
    ax.bar(x + width, f_score, width=width, label='F Score')

    ax.set_ylim(0.99, 1)

    ax.set_xlabel('Modelos')
    ax.set_title('Métricas de evaluación')

    plt.xticks(x, models)
    plt.legend(loc='best')
    plt.show()

