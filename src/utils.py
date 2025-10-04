import os
import sys
import dill # Used for saving/loading complex Python objects like pipelines
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Saves a Python object (like a ColumnTransformer or model) to a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Use 'dill' for robust saving of complex scikit-learn pipelines
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a Python object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)