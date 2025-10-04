import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Creates the data transformation pipeline for the P2P Credit Risk Data (11 Features)."""
        try:
            # --- Feature Lists (Matching the 12 Columns exactly) ---
            
            # Numerical features: (Needs Imputation for NaN values + Scaling)
            numerical_features = [
                'person_age', 
                'person_income', 
                'person_emp_length',  
                'loan_amnt', 
                'loan_int_rate', 
                'loan_percent_income',
                'cb_person_cred_hist_length' 
            ] 
            
            # Categorical features: (Needs Imputation for mode + OneHot Encoding)
            nominal_features = [
                'person_home_ownership', 
                'loan_intent', 
                'loan_grade', 
                'cb_person_default_on_file'
            ]
            
            logging.info(f"Numerical features to be scaled: {numerical_features}")
            logging.info(f"Categorical features to be encoded: {nominal_features}")
            
            # --- Build Pipelines ---
            
            # 1. Numerical Pipeline: Impute missing values (median) + Scale/Standardize
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Handles NaNs in emp_length, int_rate
                ('scaler', StandardScaler())
            ])
            
            # 2. Categorical Pipeline: Impute missing (mode) + OneHot Encoding
            nominal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # --- Combine All Pipelines ---
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("nominal_pipeline", nominal_pipeline, nominal_features),
                ],
                remainder='drop',
                n_jobs=-1
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Loads data, fits the pipeline, transforms data, and saves the pipeline object."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data for transformation.")
            preprocessing_obj = self.get_data_transformer_object()

            # Target column: 'loan_status' (0 or 1)
            target_column_name = 'loan_status'
            
            # Separate features (X) and target (y)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Fit and Transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target back into arrays for the model trainer
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessing object (pipeline).")
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