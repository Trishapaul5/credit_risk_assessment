import os
import sys
import dill 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.exception import CustomException
from src.logger import logging

# --- File Handling Functions ---

def save_object(file_path: str, obj):
    """Saves a Python object as a .pkl file using the dill library."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """Loads a Python object from a .pkl file using the dill library."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
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