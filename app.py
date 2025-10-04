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
                value = request.form.get(html_key)
                
                # Convert inputs to required types
                if class_key in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                    form_data[class_key] = int(value) if value else 0
                elif class_key in ['person_emp_length', 'loan_int_rate', 'loan_percent_income']:
                    form_data[class_key] = float(value) if value else 0.0
                else:
                    form_data[class_key] = value

            # 2. Instantiate CustomData with the 11 required arguments (kwargs)
            data = CustomData(**form_data)
            
            # Convert captured data into a DataFrame
            pred_df = data.get_data_as_dataframe()

            # 3. Run Prediction Pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df) # Returns [0] or [1]
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
            error_msg = f"System Error: {e}"
            logging.error(error_msg, exc_info=True)
            return render_template('home.html', results="System Error: An unexpected error occurred. Please check server logs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)