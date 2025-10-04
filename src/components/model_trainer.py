import os
import sys
import numpy as np
from dataclasses import dataclass

# Import ML Algorithms
from sklearn.linear_model import LogisticRegression 
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
            models = {
                # FIX: Changed 'Logisticેશન' to 'LogisticRegression'
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000), 
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
            # Added max_iter=1000 to Logistic Regression to prevent convergence warnings/errors
            raise CustomException(e, sys)