from codecarbon import EmissionsTracker
from eco2ai import Tracker
import os

def track_training_codecarbon(model, X_train, y_train):
    """Tracks model's training process with codecarbon"""

    tracker = EmissionsTracker()

    tracker.start()
    model.fit(X_train, y_train)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")

def track_training_eco2ai(model, X_train, y_train):
    """Tracks model's training process with eco2ai"""

    tracker = Tracker(
        project_name="TFG_Project",
        experiment_description="training",
        alpha_2_code="ES-MD"
    )

    tracker.start()
    model.fit(X_train, y_train)
    tracker.stop()

def track_benchmarking_codecabon(estimators_list, X_train, y_train):
    """Tracks benchmarking's training process with codecarbon"""

    if os.path.exists("emissions.csv"):
        os.remove("emissions.csv")

    for estimator in estimators_list:
        track_training_codecarbon(estimator, X_train, y_train)


def track_benchmarking_eco2ai(estimators_list, X_train, y_train):
    """Tracks benchmarking's training process with eco2ai"""

    if os.path.exists("emission.csv"):
        os.remove("emission.csv")

    for estimator in estimators_list:
        track_training_eco2ai(estimator, X_train, y_train)