from codecarbon import EmissionsTracker
from eco2ai import Tracker

def track_training_codecarbon(model, X_train, y_train):
    print('modelo: ', model)

    tracker = EmissionsTracker()

    tracker.start()
    model.fit(X_train, y_train)
    emissions: float = tracker.stop()
    print(f"Emissions: {emissions} kg")

def track_training_eco2ai(model, X_train, y_train):
    print('modelo: ', model)

    tracker = Tracker(
        project_name="TFG_Project",
        experiment_description="training",
        alpha_2_code="ES-MD"
    )

    tracker.start()
    model.fit(X_train, y_train)
    tracker.stop()
