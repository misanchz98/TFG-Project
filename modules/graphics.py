import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(model, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


def boxplot_df(df):
    """Boxplot all Dataframe"""
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, orient="h")