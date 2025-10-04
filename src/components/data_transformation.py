import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer # To create the pipeline [cite: 282]
from sklearn.impute import SimpleImputer # To handle missing values [cite: 283]
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # Import the save helper

@dataclass
class DataTransformationConfig:
    """Configuration class for Data Transformation component."""
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates the data transformation pipeline (ColumnTransformer).
        This will be saved as a pickle file for later use.
        """
        try:
            # --- 1. Define Feature Types (Based on German Credit Data EDA) ---
            # Numerical features usually require scaling/imputation
            numerical_features = ['Duration_in_months', 'Credit_amount', 'Age_in_years', 'Installment_rate_in_percentage_of_disposable_income', 'Present_residence_since'] 
            
            # Categorical features requiring standard OneHot Encoding
            nominal_features = ['Purpose', 'Other_debtors_or_guarantors', 'Housing']

            # Ordinal/Domain features for Target-Guided Encoding
            # These are ordered based on risk/domain knowledge (like credit history)
            # The order must be determined during initial EDA (e.g., in notebooks/EDA.ipynb)
            ordinal_features = ['Checking_account_status', 'Credit_history', 'Savings_account_or_bonds', 'Present_employment_since']

            # --- 2. Pipeline for Numerical Features (Standardization) ---
            num_pipeline = ColumnTransformer(
                [
                    ("StandardScaler", StandardScaler(), numerical_features),
                    # Impute missing values with median for robustness against outliers [cite: 295, 296]
                    ("SimpleImputer", SimpleImputer(strategy='median'), numerical_features)
                ],
                remainder='passthrough'
            )
            
            # --- 3. Pipeline for Nominal Categorical Features (OneHot) ---
            nominal_pipeline = OneHotEncoder(handle_unknown='ignore')
            
            # --- 4. Pipeline for Ordinal/Domain Features (Target-Guided/Custom Encoding) ---
            # NOTE: For a true WoE/Target-Guided approach, you would pre-calculate the WOE values 
            # in the EDA phase and use a custom transformer here. 
            # For simplicity, we'll simulate the ordinal transformation using a base OrdinalEncoder 
            # that you will manually provide the categories for after your EDA.
            ordinal_pipeline = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

            # --- 5. Combine All Pipelines using ColumnTransformer [cite: 282] ---
            preprocessor = ColumnTransformer(
                [
                    ("Num_Pipeline", num_pipeline, numerical_features),
                    ("Nominal_Pipeline", nominal_pipeline, nominal_features),
                    ("Ordinal_Pipeline", ordinal_pipeline, ordinal_features)
                ],
                # Do not drop unlisted columns
                n_jobs=-1
            )
            
            logging.info("Numerical columns processed (Scaling/Imputation).")
            logging.info("Categorical columns processed (OneHot Encoding).")
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Loads data, fits the pipeline, transforms data, and saves the pipeline object."""
        try:
            # Load the train and test data ingested from the previous step
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data for transformation.")

            # Identify target and feature columns
            # Target column: Risk (1=Bad Credit/Default, 0=Good Credit)
            target_column_name = 'Risk'
            
            # Separate features (X) and target (y)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Get the transformation object (pipeline)
            preprocessing_obj = self.get_data_transformer_object()

            # --- Fit and Transform the Data ---
            logging.info("Applying preprocessing object on training and testing data.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target back into arrays for the model trainer
            # Note: We must stack the target array as an array of the same type/dimension
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessing object (pipeline).")
            
            # Save the entire transformation pipeline for deployment [cite: 365]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

# --- Integration with Data Ingestion (for testing the full pipeline) ---
# NOTE: This part belongs in a pipeline/training_pipeline.py file, but is shown here for context.
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # 1. Run Data Ingestion
    # obj_ingestion = DataIngestion()
    # train_data_path, test_data_path = obj_ingestion.initiate_data_ingestion('data/german_credit.csv')
    
    # 2. Run Data Transformation
    # obj_transformation = DataTransformation()
    # train_arr, test_arr, preprocessor_path = obj_transformation.initiate_data_transformation(
    #     train_data_path, test_data_path
    # )
    # logging.info(f"Preprocessor saved at: {preprocessor_path}")
    # logging.info(f"Transformed Train Array Shape: {train_arr.shape}")
    pass