from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os
import traceback

# NOTE: Ensure you have fixed logger.py to include sys.stdout handler for AWS EB visibility
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

# CRITICAL FIX 1: Corrected Flask application magic variable
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

                # Handle categorical features
                if class_key in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
                    # Ensure categorical values are stripped or explicitly None
                    form_data[class_key] = raw_value.strip() if raw_value else None
                    if form_data[class_key] is None:
                         # CRITICAL: Raise error for missing categorical field
                        raise ValueError(f"Missing required categorical input: {html_key}")
                    continue

                # Handle numeric features
                safe_value = None
                if raw_value and raw_value.strip() != "":
                    try:
                        if class_key in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                            safe_value = int(float(raw_value.strip()))
                        elif class_key in ['person_emp_length', 'loan_int_rate', 'loan_percent_income']:
                            safe_value = float(raw_value.strip())
                    except ValueError:
                        # Type conversion failed
                        pass 

                # CRITICAL FIX 2: Instead of defaulting to 0/0.0, raise an error if the value is missing or invalid.
                if safe_value is None:
                    raise ValueError(f"Invalid or missing required numeric input: {html_key}. Received '{raw_value}'.")
                
                form_data[class_key] = safe_value


            # 2. Instantiate CustomData and convert to DataFrame
            # Note: CustomData validation relies on the error checks above to ensure no Nones or bad types reach it.
            data = CustomData(**form_data)
            pred_df = data.get_data_as_dataframe()
            logging.info(f"Input Data for Prediction (Cleaned):\n{pred_df.to_dict()}")

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
            # --- Enhanced Error Debugging ---
            tb_str = traceback.format_exc()
            # Log the full traceback to the custom logger (and stdout, if logger.py is fixed)
            logging.error(f"Prediction failed with error:\n{tb_str}") 
            return render_template(
                'home.html',
                # Display the user-friendly error message from ValueError/CustomException
                results=f"Application Error: {str(e)}", 
                prediction_status=tb_str  # full traceback for internal debugging
            )

# CRITICAL FIX 1: Corrected __main__ magic variable
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
