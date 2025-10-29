import os


# Example parameters (these would typically come from a config file or command line arguments)
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}
MODEL_PATH = os.path.join("models", "model.joblib")
