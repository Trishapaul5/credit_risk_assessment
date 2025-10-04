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
        
        # 1. Data Ingestion
        ingestion = DataIngestion()
        raw_data_path = 'data/credit_risk_data.csv' 
        train_path, test_path = ingestion.initiate_data_ingestion(raw_data_path)
        logging.info("Data Ingestion completed.")
        
        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data Transformation completed. Preprocessor saved.")
        
        # 3. Model Trainer
        trainer = ModelTrainer()
        best_model_name, best_cost = trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"--- END-TO-END TRAINING PIPELINE SUCCESSFUL ---")
        logging.info(f"Best Model: {best_model_name}, Final Misclassification Cost: {best_cost}")

    except CustomException as e:
        logging.error(f"Pipeline failed (Custom Error): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)