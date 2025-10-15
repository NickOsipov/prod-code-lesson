import joblib


def load_model(model_path: str):
    """
    Loads a machine learning model from the specified path.

    Parameters
    ----------
    model_path : str
        The file path from where the model should be loaded.

    Returns
    -------
    The loaded machine learning model.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at the specified path: {model_path}"
        )
    

def predict(model, X):
    """
    Makes predictions using the provided model and feature set.

    Parameters
    ----------
    model : The machine learning model to be used for predictions.
    X : pd.DataFrame
        The feature set for making predictions.

    Returns
    -------
    np.ndarray
        The predictions made by the model.
    """
    return model.predict(X)
