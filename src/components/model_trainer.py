import os
import sys
import numpy as np
from dataclasses import dataclass

# Import ML Algorithms
from sklearn.linear_model import LogisticRegression # The financial industry standard
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models # Import helpers

@dataclass
class ModelTrainerConfig:
    """Stores the path where the final best model will be saved."""
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Splits transformed data, trains and evaluates models, and saves the best model.
        """
        try:
            logging.info("Splitting training and test input data.")
            # Assume last column of array is the target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            # Define Model Dictionary
            # Logistic Regression is essential due to its interpretability (used in Basel II/III)
            models = {
                "Logistic Regression": Logisticેશન(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            }

            # Evaluate Models
            model_report:dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            # Find the best model based on the LOWEST Total Cost
            best_model_score = float('inf') # Start with a high value for cost minimization
            best_model_name = ""
            best_model = None

            for name, metrics in model_report.items():
                cost = metrics['Total Cost']
                logging.info(f"Model {name}: Total Misclassification Cost: {cost}")
                
                if cost < best_model_score:
                    best_model_score = cost
                    best_model_name = name
                    best_model = models[name]

            # Check for a reasonable threshold
            # Since Total Cost is dataset-dependent, we only check if a model was found
            if best_model_name == "":
                raise CustomException("No suitable model found (Check data or cost matrix setup)", sys)

            logging.info(f"Best model found: {best_model_name} with Total Cost: {best_model_score}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

# --- Full Pipeline Execution (Requires DataIngestion and DataTransformation) ---
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # Run all components sequentially to test the full pipeline:
    # try:
    #     ingestion = DataIngestion()
    #     train_path, test_path = ingestion.initiate_data_ingestion('data/german_credit.csv')
        
    #     transformation = DataTransformation()
    #     train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
    #         train_path, test_path
    #     )
        
    #     trainer = ModelTrainer()
    #     best_model_name, best_cost = trainer.initiate_model_trainer(train_arr, test_arr)
        
    #     logging.info(f"--- END-TO-END PIPELINE SUCCESSFUL ---")
    #     logging.info(f"Best Model: {best_model_name}, Final Cost: {best_cost}")
        
    # except Exception as e:
    #     logging.error(f"Pipeline failed: {e}")
    #     pass
    pass