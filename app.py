from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# --- FIELD MAPPING for 11 Features ---
FIELD_MAPPING = {
    'age': 'person_age',
    'income': 'person_income',
    'ownership': 'person_home_ownership',
    'emp_length': 'person_emp_length',
    'intent': 'loan_intent',
    'grade': 'loan_grade',
    'loan_amount': 'loan_amnt',
    'int_rate': 'loan_int_rate',
    'percent_income': 'loan_percent_income',
    'default_on_file': 'cb_person_default_on_file',
    'cred_hist_length': 'cb_person_cred_hist_length'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # 1. Capture Data from the Form
            form_data = {}
            for html_key, class_key in FIELD_MAPPING.items():
                raw_value = request.form.get(html_key)
                
                # --- ROBUST INPUT HANDLING (Final Fail-Proof Logic) ---
                
                # Check 1: Handle Categorical Fields (strings, select options)
                if class_key in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
                    # Pass the string directly. If it's missing, CustomData will receive None.
                    form_data[class_key] = raw_value.strip() if raw_value else None
                    continue # Move to the next field

                # Check 2: Handle Numerical Fields (int/float)
                safe_value = None
                if raw_value and raw_value.strip() != "":
                    try:
                        if class_key in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                            # Attempt to convert to Integer
                            safe_value = int(float(raw_value.strip())) # Convert float first to handle '1.0'
                        elif class_key in ['person_emp_length', 'loan_int_rate', 'loan_percent_income']:
                            # Attempt to convert to Float
                            safe_value = float(raw_value.strip())
                    except ValueError:
                        # CRASH PROTECTION: If user types "abc" or "10,000", safe_value remains None
                        safe_value = None 
                
                # Assign the safe value (0 or 0.0 will be passed to CustomData if non-numeric or missing)
                if class_key in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                    form_data[class_key] = safe_value if safe_value is not None else 0 
                elif class_key in ['person_emp_length', 'loan_int_rate', 'loan_percent_income']:
                    form_data[class_key] = safe_value if safe_value is not None else 0.0

            # 2. Instantiate CustomData 
            data = CustomData(**form_data)
            pred_df = data.get_data_as_dataframe()
            
            logging.info(f"Input Data for Prediction (Cleaned):\n{pred_df}")

            # 3. Run Prediction Pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            prediction = results[0]

            # 4. Financial Interpretation
            if prediction == 1:
                recommendation = "REJECT LOAN: High Risk of Default (PD = 1)"
                cost_impact = "Warning: Approving this loan carries a significant **5x Misclassification Cost** (False Negative risk)."
            else:
                recommendation = "APPROVE LOAN: Low Risk of Default (PD = 0)"
                cost_impact = "Expected Loss is minimized. Recommendation aligns with **1x Misclassification Cost** (False Positive risk)."

            # 5. Render Results
            return render_template('home.html', 
                                   results=recommendation,
                                   prediction_status=cost_impact)

        except Exception as e:
            # Catch exceptions that CustomData or the Pipeline might throw (e.g., missing artifact)
            error_msg = f"Prediction failed due to unhandled error: {e}"
            logging.error(error_msg, exc_info=True)
            return render_template('home.html', results=f"Application Error: {e.__class__.__name__}. Check server logs for detail.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
