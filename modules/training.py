from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from codecarbon import EmissionsTracker
from modules.preprocess import split_dataset, get_features, get_output
from cumulator import base


# CODECARBON
def train_LR_codecarbon(df, train_size=0.25):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    train_features = get_features(train_df)
    print(train_features)
    y = get_output(train_df)
    print(y)

    # Step 3: train LogisticRegression model
    # lg_pipeline = Pipeline([("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
    # tracker = EmissionsTracker()

    # tracker.start()
    # lg_pipeline.fit(X, y)
    # emissions: float = tracker.stop()
    # print(f"Emissions: {emissions} kg")


def train_RF_codecarbon(train_features, train_output, n_estimators=100, max_leaf_nodes=16, n_jobs=-1):
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)
    tracker = EmissionsTracker()

    tracker.start()
    rnd_clf.fit(train_features, train_output)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")


def train_SVC_codecarbon(train_features, train_output):
    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    tracker = EmissionsTracker()

    tracker.start()
    svm_pipeline.fit(train_features, train_output)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")


# CUMULATOR
def train_LR_cumulator(train_features, train_output):
    lg_pipeline = Pipeline([("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
    cumulator = base.Cumulator()
    cumulator.run(lg_pipeline.fit, X=train_features, y=train_output)
    print(cumulator.total_carbon_footprint())
    cumulator.display_carbon_footprint()


def train_RF_cumulator(train_features, train_output, n_estimators=100, max_leaf_nodes=16, n_jobs=-1):
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)
    cumulator = base.Cumulator()
    cumulator.run(rnd_clf.fit, X=train_features, y=train_output)
    print(cumulator.total_carbon_footprint())
    cumulator.display_carbon_footprint()


def train_SVC_cumulator(train_features, train_output):
    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    cumulator = base.Cumulator()
    cumulator.run(svm_pipeline.fit, X=train_features, y=train_output)
    print(cumulator.total_carbon_footprint())
    cumulator.display_carbon_footprint()
