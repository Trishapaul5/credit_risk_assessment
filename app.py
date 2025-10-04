from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys
import os

# Initialize the Flask application
application = Flask(__name__)
app = application

# Define the main prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If GET, render the input form (you'll create this later)
        return render_template('home.html')
    else:
        try:
            # 1. Capture Data from the Form/API
            # Create a CustomData object by reading input fields
            data = CustomData(
                # NOTE: Replace 'request.form.get' keys with your actual HTML/API field names
                Status_of_existing_checking_account=request.form.get('checking_status'),
                Duration_in_month=int(request.form.get('duration')),
                Credit_history=request.form.get('credit_history'),
                Purpose=request.form.get('purpose'),
                Credit_amount=int(request.form.get('credit_amount')),
                # ... [Capture the remaining features] ...
                Age_in_years=int(request.form.get('age')),
                Foreign_worker=request.form.get('foreign_worker')
            )
            
            # Convert captured data into a DataFrame
            pred_df = data.get_data_as_dataframe()

            # 2. Run Prediction Pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df) # This returns [0] or [1]

            # 3. Financial Interpretation
            prediction = results[0]
            
            if prediction == 1:
                # Predicted Bad Credit (Default)
                recommendation = "Reject Loan: High Probability of Default (PD)."
                cost_impact = "Risk of incurring a 5x financial loss (False Negative penalty)."
            else:
                # Predicted Good Credit (No Default)
                recommendation = "Approve Loan: Low Probability of Default (PD)."
                cost_impact = "Low risk of loss; minimizes 1x False Positive penalty."

            # 4. Render Results
            return render_template('home.html', 
                                   results=recommendation,
                                   prediction_status=cost_impact)

        except Exception as e:
            # Log and handle any runtime errors
            error_message = f"Prediction failed: {e}"
            return render_template('home.html', results=error_message)

if __name__ == "__main__":
    # In a deployment environment (like AWS/Azure), you'll typically use gunicorn
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    app.run(host="0.0.0.0", debug=True)