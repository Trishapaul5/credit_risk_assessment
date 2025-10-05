import os
import sys
import joblib  # CRITICAL: Replace dill with joblib for ML artifacts and compression
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.exception import CustomException
from src.logger import logging

# --- File Handling Functions ---

def save_object(file_path: str, obj):
    """Saves a Python object as a .pkl file using joblib (with compression)."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Use joblib's native compression, level 3 is a good balance of speed/ratio.
        # This replaces the old dill/open logic.
        joblib.dump(obj, file_path, compress=('gzip', 3))

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Loads a Python object from a .pkl file using joblib."""
    try:
        # joblib automatically handles reading compressed files
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)


# --- Financial Evaluation Function ---

def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains models and evaluates based on the Total Misclassification Cost (5:1 penalty).
    Returns a report keyed by model name.
    """
    try:
        report = {}
        for model_name, model in models.items():
            
            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)
            
            # Calculate Confusion Matrix: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_test, y_test_pred)
            TN, FP, FN, TP = cm.ravel()
            
            # Total Misclassification Cost = (Cost of FN * FN) + (Cost of FP * FP)
            # Cost = (5 * FN) + (1 * FP)
            TOTAL_MISCLASSIFICATION_COST = (5 * FN) + (1 * FP)
            
            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            
            report[model_name] = {
                'Accuracy': test_model_accuracy,
                'Total Cost': TOTAL_MISCLASSIFICATION_COST,
                'Confusion Matrix': cm.tolist()
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
