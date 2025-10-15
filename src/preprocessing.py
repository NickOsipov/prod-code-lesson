"""
Module: preprocessing.py
Description: This module contains a function to sum up elements in an array.
"""

from typing import Optional

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> tuple:
    """
    Load and return the Iris dataset.

    Returns
    -------
    df : DataFrame
        A pandas DataFrame containing the Iris dataset.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
) -> tuple:
    """
    Split the DataFrame into training and testing sets.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame to be split.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
    stratify : Series, optional
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns
    -------
    train : DataFrame
        Training features.
    test : DataFrame
        Testing features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1),
        df["target"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test

def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the DataFrame to a Parquet file.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be saved.
    path : str
        The file path where the Parquet file will be saved.
    """
    df.to_parquet(path, index=False)


def preprocess_data() -> tuple:
    """
    Load, split, and save the Iris dataset.
    """
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, stratify=df["target"])

    return X_train, y_train, X_test, y_test
