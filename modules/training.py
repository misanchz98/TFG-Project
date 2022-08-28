from codecarbon import EmissionsTracker
from eco2ai import Tracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from carbontracker.tracker import CarbonTracker
from modules.preprocess import split_dataset, get_features, get_output


# CODECARBON
def train_LR_codecarbon(df, train_size=0.25):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train LogisticRegression model and track with codecarbon
    lg_pipeline = Pipeline([("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
    tracker = EmissionsTracker()

    tracker.start()
    lg_pipeline.fit(X, y)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")


def train_RF_codecarbon(df, train_size=0.25, n_estimators=100, max_leaf_nodes=16, n_jobs=-1):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train RandomForest model and track with codecarbon
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)
    tracker = EmissionsTracker()

    tracker.start()
    rnd_clf.fit(X, y)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")


def train_SVC_codecarbon(df, train_size=0.25):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train SVM model and track with codecarbon
    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    tracker = EmissionsTracker()

    tracker.start()
    svm_pipeline.fit(X, y)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")


# ECO2AI
def train_LR_eco2ai(df, train_size=0.25):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train LogisticRegression model and track with eco2ai
    lg_pipeline = Pipeline([("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
    tracker = Tracker(
        project_name="TFG_Project",
        experiment_description="training LogisticRegression model",
        file_name="eco2ai_emissions.csv"
    )

    tracker.start()
    lg_pipeline.fit(X, y)
    tracker.stop()


def train_RF_eco2ai(df, train_size=0.25, n_estimators=100, max_leaf_nodes=16, n_jobs=-1):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train RandomForest model and track with eco2ai
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)
    tracker = Tracker(
        project_name="TFG_Project",
        experiment_description="training RandomForest model",
        file_name="eco2ai_emissions.csv"
    )

    tracker.start()
    rnd_clf.fit(X, y)
    tracker.stop()


def train_SVC_eco2ai(df, train_size=0.25):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train SVM model and track with eco2ai
    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    tracker = Tracker(
        project_name="TFG_Project",
        experiment_description="training LogisticRegression model",
        file_name="eco2ai_emissions.csv"
    )

    tracker.start()
    svm_pipeline.fit(X, y)
    tracker.stop()

#  CARBONTRACKER

def train_LR_carbontracker(df, train_size=0.25, max_epochs=1):
    # Step 1: split dataset into training and test
    train_df, test_df = split_dataset(df, train_size=train_size)

    # Step 2: split training into features and output
    X = get_features(train_df)
    y = get_output(train_df)

    # Step 3: train LogisticRegression model and track with carbontracker
    lg_pipeline = Pipeline([("scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
    tracker = CarbonTracker(epochs=max_epochs)

    for epoch in range(max_epochs):
        tracker.epoch_start()
        lg_pipeline.fit(X, y)
        tracker.epoch_end()

    tracker.stop()
