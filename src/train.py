import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    model: RandomForestClassifier, 
    model_params: dict,
    X: pd.DataFrame, 
    y: pd.Series
) -> RandomForestClassifier:
    """
    Trains the provided model using the given features and target.

    Parameters
    ----------
    model : RandomForestClassifier
        The machine learning model to be trained.
    X : pd.DataFrame
        The feature set.
    y : pd.Series
        The target variable.

    Returns
    -------
    RandomForestClassifier
        The trained model.
    """

    model_instance = model(**model_params)
    model_instance.fit(X, y)
    return model_instance


def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Saves the trained model to the specified path.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained machine learning model.
    model_path : str
        The file path where the model should be saved.
    """
    joblib.dump(model, model_path)


def train_pipeline(
    model: RandomForestClassifier, 
    model_params: dict,
    X: pd.DataFrame, 
    y: pd.Series, 
    model_path: str
) -> None:
    """
    Trains the model and saves it to the specified path.

    Parameters
    ----------
    model : RandomForestClassifier
        The machine learning model to be trained.
    X : pd.DataFrame
        The feature set.
    y : pd.Series
        The target variable.
    model_path : str
        The file path where the model should be saved.
    """
    trained_model = train_model(model, model_params, X, y)
    save_model(trained_model, model_path)
