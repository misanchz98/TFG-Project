from sklearn.metrics import precision_score, recall_score, f1_score


def get_precision(y_test, y_pred):
    return precision_score(y_test, y_pred)


def get_recall(y_test, y_pred):
    return recall_score(y_test, y_pred)


def get_f_score(y_test, y_pred):
    return f1_score(y_test, y_pred)


def get_classification_metrics(y_pred, y_test):
    precision = get_precision(y_test, y_pred)
    recall = get_recall(y_test, y_pred)
    f_score = get_f_score(y_test, y_pred)

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f_score)

    return precision, recall, f_score


