"""
Script: pipeline.py
Description: This script contains functions to create a machine learning pipeline that includes data preprocessing, model training
"""

from sklearn.ensemble import RandomForestClassifier
from loguru import logger

from src.train import train_model, save_model
from src.inference import predict, load_model
from src.preprocessing import preprocess_data
from src.evaluate import evaluate_model
from config.variables import MODEL_PARAMS, MODEL_PATH


def main():
    """
    Main function to run the machine learning pipeline.
    """
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = preprocess_data()
    
    # Train the model
    logger.info("Training the model...")
    model = train_model(RandomForestClassifier, MODEL_PARAMS, X_train, y_train)
    
    # Save the trained model
    logger.info(f"Saving the model to {MODEL_PATH}...")
    save_model(model, MODEL_PATH)

    # Load the model for inference
    logger.info(f"Loading the model from {MODEL_PATH}...")
    loaded_model = load_model(MODEL_PATH)
    
    # Make predictions
    logger.info("Making predictions on the test set...")
    predictions = predict(loaded_model, X_test)
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluation_metrics = evaluate_model(predictions, y_test)
    
    logger.info(f"Evaluation Metrics: {evaluation_metrics}")


if __name__ == "__main__":
    main()
