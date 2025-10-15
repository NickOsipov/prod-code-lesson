import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(y_pred: pd.Series, y_test: pd.Series) -> dict:
    """
    Evaluates the provided model using the test features and target.

    Parameters
    ----------
    y_pred : pd.Series
        The predicted values from the model.
    y_test : pd.Series
        The true target values for testing.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics such as accuracy and classification report.
    """
    accuracy = accuracy_score(y_test, y_pred)

    return {"accuracy": accuracy}
