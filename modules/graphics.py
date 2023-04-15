import matplotlib.pyplot as plt
from modules import preprocess
import seaborn as sns

def boxplot_s_temp(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    ax1.boxplot(df['S1_Temp'])
    ax1.set_title('S1_Temp')

    ax2.boxplot(df['S2_Temp'])
    ax2.set_title('S2_Temp')

    ax3.boxplot(df['S3_Temp'])
    ax3.set_title('S3_Temp')

    ax4.boxplot(df['S4_Temp'])
    ax4.set_title('S4_Temp')

    fig.suptitle('Boxplots para Si_Temp')
    fig.tight_layout()

    plt.show()

def boxplot_s_light(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    ax1.boxplot(df['S1_Light'])
    ax1.set_title('S1_Light')

    ax2.boxplot(df['S2_Light'])
    ax2.set_title('S2_Light')

    ax3.boxplot(df['S3_Light'])
    ax3.set_title('S3_Light')

    ax4.boxplot(df['S4_Light'])
    ax4.set_title('S4_Light')

    fig.suptitle('Boxplots para Si_Light')
    fig.tight_layout()

    plt.show()

def boxplot_s_sound(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    ax1.boxplot(df['S1_Sound'])
    ax1.set_title('S1_Sound')

    ax2.boxplot(df['S2_Sound'])
    ax2.set_title('S2_Sound')

    ax3.boxplot(df['S3_Sound'])
    ax3.set_title('S3_Sound')

    ax4.boxplot(df['S4_Sound'])
    ax4.set_title('S4_Sound')

    fig.suptitle('Boxplots para Si_Sound')
    fig.tight_layout()

    plt.show()

def boxplot_s_co2(df):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

    ax1.boxplot(df['S5_CO2'])
    ax1.set_title('S5_CO2')

    ax2.boxplot(df['S5_CO2_Slope'])
    ax2.set_title('S5_CO2_Slope')

    fig.suptitle('Boxplots para S5_CO2, S5_CO2_Slope')
    fig.tight_layout()

    plt.show()

def boxplot_s_pir(df):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

    ax1.boxplot(df['S6_PIR'])
    ax1.set_title('S6_PIR')

    ax2.boxplot(df['S7_PIR'])
    ax2.set_title('S7_PIR')

    fig.suptitle('Boxplots para Si_PIR')
    fig.tight_layout()

    plt.show()

def boxplot_df(df):
    """Boxplot all Dataframe"""
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, orient="h")