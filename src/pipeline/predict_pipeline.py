import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # Helper to load the saved .pkl files
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads preprocessor and model, transforms features, and predicts the outcome.
        :param features: A DataFrame containing new applicant data.
        :return: Prediction array (0 or 1).
        """
        try:
            # Define paths (must match ModelTrainer/DataTransformation configurations)
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
            
            # Load the saved objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform the new data using the fitted preprocessor
            data_scaled = preprocessor.transform(features)

            # Make the prediction
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Class to map raw input data (from a web form/API) into a DataFrame suitable for the model.
    """
    def __init__(self, 
                 Status_of_existing_checking_account: str,
                 Duration_in_month: int,
                 Credit_history: str,
                 Purpose: str,
                 Credit_amount: int,
                 Savings_account_or_bonds: str,
                 Present_employment_since: str,
                 Installment_rate_in_percentage_of_disposable_income: int,
                 Personal_status_and_sex: str,
                 Other_debtors_or_guarantors: str,
                 Present_residence_since: int,
                 Property: str,
                 Age_in_years: int,
                 Other_installment_plans: str,
                 Housing: str,
                 Number_of_existing_credits_at_this_bank: int,
                 Job: str,
                 Number_of_people_being_liable_to_provide_maintenance_for: int,
                 Telephone: str,
                 Foreign_worker: str
                 ):
        # Map all features to instance variables
        self.Status_of_existing_checking_account = Status_of_existing_checking_account
        self.Duration_in_month = Duration_in_month
        self.Credit_history = Credit_history
        self.Purpose = Purpose
        self.Credit_amount = Credit_amount
        # ... [Continue mapping all 20 features from the German Credit Data] ...
        self.Housing = Housing
        self.Age_in_years = Age_in_years
        self.Foreign_worker = Foreign_worker

    def get_data_as_dataframe(self):
        """Converts user input variables into a single Pandas DataFrame."""
        try:
            custom_data_input_dict = {
                'Status_of_existing_checking_account': [self.Status_of_existing_checking_account],
                'Duration_in_month': [self.Duration_in_month],
                'Credit_history': [self.Credit_history],
                # ... [Map the remaining 17 features] ...
                'Age_in_years': [self.Age_in_years],
                'Foreign_worker': [self.Foreign_worker]
            }
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)

# NOTE: For the final implementation, ensure ALL 20 columns are mapped in __init__ and get_data_as_dataframe.