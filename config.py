"""
Configuration settings for Explainable AI Loan Default Prediction application.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'loan_data.csv')

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'loan_default_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, 'feature_columns.pkl')
CATEGORICAL_MAPPINGS_PATH = os.path.join(MODELS_DIR, 'categorical_mappings.pkl')

# Database
DB_PATH = os.path.join(BASE_DIR, 'instance', 'predictions.db')

# Flask Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'explainable-ai-loan-default-secret-key-2025')
DEBUG = True
PORT = 5000
HOST = '127.0.0.1'

# Feature Schema
FEATURE_COLUMNS = [
    'Age', 'Number_of_Dependents', 'Employment_Experience',
    'Monthly_Income', 'Annual_Income', 'Credit_Score',
    'Loan_Amount', 'Loan_Term', 'Interest_Rate', 'Existing_Loans',
    'Gender_encoded', 'Marital_Status_encoded',
    'Employment_Type_encoded', 'Loan_Purpose_encoded'
]

CATEGORICAL_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'Marital_Status': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'Employment_Type': {'Salaried': 0, 'Self-Employed': 1, 'Freelancer': 2, 'Unemployed': 3},
    'Loan_Purpose': {'Home': 0, 'Education': 1, 'Personal': 2, 'Business': 3, 'Vehicle': 4}
}

# Risk Thresholds
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40
