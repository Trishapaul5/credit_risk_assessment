💰 End-to-End MLOps Project: Cost-Optimized Credit Risk Assessment
This project demonstrates a production-grade MLOps pipeline for a Probability of Default (PD) model used in P2P (Peer-to-Peer) lending. The core focus is on aligning the Machine Learning metric directly with business objectives by minimizing the Total Misclassification Cost.

The deployed application runs a Flask API, containerized via Docker and hosted on AWS Elastic Beanstalk.

🚀 Live Application URL
The final application, running on AWS, can be accessed here:
Credit Risk Assessment App

✨ Unique Value Proposition
The model selection process is explicitly optimized for financial impact rather than standard accuracy.

Misclassification Type

Financial Cost Weight

Rationale

False Negative (FN)

5x

Classifying a bad customer (who defaults) as "Good." This results in a 5× loss of principal, interest, and operational expense.

False Positive (FP)

1x

Classifying a good customer (who repays) as "Bad." This results in a 1× lost opportunity cost.

The best model is selected by minimizing the custom metric: (5×FN)+(1×FP).

🏗️ Project Architecture Overview
The project follows a modular, MLOps structure to separate concerns and ensure reproducibility.

credit_risk_assessment/
├ ── artifacts/                # Trained model & preprocessor artifacts (.pkl files)
├ ── data/                     # Raw dataset (.csv)
├ ── logs/                     # Runtime and training logs
│
├ ── src/                      # Primary Python Package (Modular Code)
│    ├ ── components/          # Data Ingestion, Transformation, Model Training
│    ├ ── pipeline/            # Training and Live Prediction Logic
│    ├ ── exception.py         # Custom error handling for detailed tracebacks
│    ├ ── logger.py            # Enhanced logger for AWS visibility (sys.stdout fix)
│    └── utils.py             # Model evaluation (Cost function) and object saving (Joblib)
│
├ ── app.py                    # Flask API Entry Point (Handles web requests and input validation)
├ ── Dockerfile                # Container blueprint (Alpine + CMake/build-base for XGBoost)
├ ── Dockerrun.aws.json        # AWS Elastic Beanstalk configuration
└── requirements.txt          # Python dependencies

🛠️ Key MLOps Techniques Demonstrated
Phase

Action Performed

MLOps/Stability Technique

Training

Model Selection (model_trainer.py)

Cost-Sensitive Evaluation to align ML metrics with business P&L.

Data Integrity

Feature engineering (data_transformation.py)

Use of fitted ColumnTransformer (preprocessor.pkl) to guarantee consistent feature scaling and encoding at inference.

API Robustness

Input Handling (app.py)

Robust validation checks for missing or non-numeric input values, preventing runtime container crashes.

Deployment

Dockerization (Dockerfile)

Used minimal Alpine base and included cmake/build-base to ensure successful, stable compilation of native dependencies (like XGBoost) on the target Linux platform.

Monitoring

Logging (logger.py)

Configured logging to pipe output to sys.stdout, ensuring logs are captured and searchable in AWS CloudWatch during live operation.

⚙️ How to Run Locally (M: Building the Model)
Prerequisites
Python 3.9+

Docker Desktop

Access to the raw data file (data/credit_risk_data.csv).

Steps
Setup Environment & Install Dependencies:

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set Python Path for package imports
export PYTHONPATH=$PWD:$PYTHONPATH 

Run the Training Pipeline:
(Ensure your dataset is placed in the data/ folder before running)

# This generates model.pkl and preprocessor.pkl in the artifacts/ folder.
python3 src/pipeline/training_pipeline.py

Run the Flask Application (Live Prediction):

# Start the API locally
python3 app.py
# Access the application at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

☁️ Deployment Commands (O: Docker & AWS)
These steps deploy the final trained model to a Docker container and push it for AWS Elastic Beanstalk consumption.

Define Tag and Build Docker Image:

DOCKER_IMAGE_TAG="trishapaul5/credit-risk-pd-model:final-deployment-v3"
docker build -t $DOCKER_IMAGE_TAG .

Push Image to Docker Hub:

docker login
docker push $DOCKER_IMAGE_TAG

Deploy to AWS Elastic Beanstalk:

# Create the deployment artifact (contains only the configuration for EB)
zip deployment_artifact.zip Dockerrun.aws.json

# Upload 'deployment_artifact.zip' to the AWS Elastic Beanstalk Console
# (Environment: CreditRiskFinalApp-env)