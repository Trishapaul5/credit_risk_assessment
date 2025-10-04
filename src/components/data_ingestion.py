import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # Used for creating class variables easily

# Import custom libraries
from src.exception import CustomException
from src.logger import logging

# --- Configuration Class ---
@dataclass
class DataIngestionConfig:
    """Stores all configuration paths for data ingestion output."""
    # Define paths for the artifacts folder (output of this component)
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    
# --- Main Ingestion Class ---
class DataIngestion:
    def __init__(self):
        # Initialize the config class to get the output paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_file_path: str):
        """
        Reads data, performs train-test split, and saves artifacts.
        
        :param data_file_path: The local path to the raw dataset (e.g., 'data/german_credit.csv').
        :return: Tuple of (train_data_path, test_data_path) for the next stage (Data Transformation).
        """
        logging.info("Entered the data ingestion method or component.")
        try:
            # 1. Data Reading (Replace with your actual file path)
            df = pd.read_csv(data_file_path)
            logging.info('Read the dataset as a DataFrame.')

            # 2. Artifacts Folder Setup
            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # 3. Save Raw Data (Best Practice)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder.")

            # 4. Train-Test Split (80/20 split)
            logging.info("Train Test Split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save Split Data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed.")
            
            # Return the paths for the next component (Data Transformation)
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Custom Exception Handling
            raise CustomException(e, sys)

# --- Execution Test ---
if __name__ == "__main__":
    # Create a temporary 'data' folder and place your downloaded CSV here
    # E.g., The German Credit Data should be saved as 'data/german_credit.csv'
    
    # 1. Download and save your German Credit Data as 'german_credit.csv' in a 'data' folder.
    # 2. Run the ingestion process:
    # obj = DataIngestion()
    # train_path, test_path = obj.initiate_data_ingestion('data/german_credit.csv')
    # logging.info(f"Train Data Path: {train_path}, Test Data Path: {test_path}")

    # You will need to replace 'data/german_credit.csv' with the actual path 
    # to the file you downloaded.
    pass