import os
import sys
from src.exception import CustomException
from src.logger import logging

# Import all core components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    try:
        logging.info("Starting End-to-End Training Pipeline for Credit Risk Model.")
        
        # --- 1. Data Ingestion ---
        # Reads the raw CSV, splits it, and saves train/test CSVs to artifacts/
        logging.info("Starting Data Ingestion component.")
        ingestion = DataIngestion()
        
        # NOTE: This path must match your new P2P dataset file location
        raw_data_path = 'data/credit_risk_data.csv' 
        train_path, test_path = ingestion.initiate_data_ingestion(raw_data_path)
        logging.info("Data Ingestion completed.")
        
        # --- 2. Data Transformation ---
        # Loads train/test data, fits the preprocessor (scaler/encoder), and saves preprocessor.pkl
        logging.info("Starting Data Transformation component.")
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data Transformation completed. Preprocessor saved.")
        
        # --- 3. Model Trainer ---
        # Trains multiple models, evaluates against the Cost Function (5:1 loss), and saves model.pkl
        logging.info("Starting Model Training component.")
        trainer = ModelTrainer()
        best_model_name, best_cost = trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"--- END-TO-END TRAINING PIPELINE SUCCESSFUL ---")
        logging.info(f"Best Model: {best_model_name}, Final Misclassification Cost: {best_cost}")

    except CustomException as e:
        # Catches custom project errors and logs them
        logging.error(f"Pipeline failed (Custom Error): {e}")
        sys.exit(1)
    except Exception as e:
        # Catches unexpected Python/System errors
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)