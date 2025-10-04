# ... [Imports and save_object/load_object functions] ...
from sklearn.metrics import accuracy_score, confusion_matrix

# ... [Existing code] ...

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains multiple models, evaluates their performance based on Total Misclassification Cost,
    and returns a report.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)
            
            # Calculate Confusion Matrix and Cost
            cm = confusion_matrix(y_test, y_test_pred)
            
            # cm is typically [[TN, FP], [FN, TP]]
            TN, FP, FN, TP = cm.ravel()
            
            # Calculate Total Misclassification Cost (5:1 penalty for FN:FP)
            # Cost = (Cost of False Negative * FN) + (Cost of False Positive * FP)
            TOTAL_MISCLASSIFICATION_COST = (5 * FN) + (1 * FP)
            
            # Calculate standard metrics for reference
            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = {
                'Accuracy': test_model_accuracy,
                'Total Cost': TOTAL_MISCLASSIFICATION_COST,
                'Confusion Matrix': cm.tolist() # Convert to list for easy logging/storage
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)