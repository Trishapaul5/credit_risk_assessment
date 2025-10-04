import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object 
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Loads preprocessor and model, transforms features, and predicts the outcome.
        :param features: A DataFrame containing new applicant data (11 features).
        :return: Prediction array (0 or 1).
        """
        try:
            # Define paths
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
            
            # Load the saved objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform the new data
            data_scaled = preprocessor.transform(features)

            # Make the prediction
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Class to map raw input data (from a web form/API) into a DataFrame with the 11 required columns.
    """
    def __init__(self, 
                 person_age: int,
                 person_income: int,
                 person_home_ownership: str,
                 person_emp_length: float,
                 loan_intent: str,
                 loan_grade: str,
                 loan_amnt: int,
                 loan_int_rate: float,
                 loan_percent_income: float,
                 cb_person_default_on_file: str,
                 cb_person_cred_hist_length: int
                 ):
        # Map all 11 features to instance variables
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_default_on_file = cb_person_default_on_file
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

    def get_data_as_dataframe(self):
        """Converts user input variables into a single Pandas DataFrame with correct column order."""
        try:
            # Keys must match the ColumnTransformer's expected input order/names
            custom_data_input_dict = {
                'person_age': [self.person_age],
                'person_income': [self.person_income],
                'person_home_ownership': [self.person_home_ownership],
                'person_emp_length': [self.person_emp_length],
                'loan_intent': [self.loan_intent],
                'loan_grade': [self.loan_grade],
                'loan_amnt': [self.loan_amnt],
                'loan_int_rate': [self.loan_int_rate],
                'loan_percent_income': [self.loan_percent_income],
                'cb_person_default_on_file': [self.cb_person_default_on_file],
                'cb_person_cred_hist_length': [self.cb_person_cred_hist_length]
            }
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)