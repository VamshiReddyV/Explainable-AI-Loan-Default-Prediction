"""
Unit tests for the loan default prediction model and preprocessing.
"""
import os
import joblib
import numpy as np
import pandas as pd
import pytest

from config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, CATEGORICAL_MAPPINGS
from src.train_model import preprocess_data, FEATURE_COLUMNS
from src.data_preprocessing import clean_dataset, encode_categorical_features


def test_model_artifacts_exist():
    """Ensure trained model files are saved and accessible."""
    assert os.path.exists(MODEL_PATH), "Model pickle missing"
    assert os.path.exists(SCALER_PATH), "Scaler pickle missing"
    assert os.path.exists(FEATURE_COLUMNS_PATH), "Feature columns pickle missing"


def test_model_prediction():
    """Verify that model produces binary predictions and calibrated probabilities."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

    # Synthetic sample
    sample_df = pd.DataFrame([{
        'Age': 35,
        'Number_of_Dependents': 1,
        'Employment_Experience': 5,
        'Monthly_Income': 60000,
        'Annual_Income': 720000,
        'Credit_Score': 750,
        'Loan_Amount': 200000,
        'Loan_Term': 36,
        'Interest_Rate': 8.5,
        'Existing_Loans': 0,
        'Gender_encoded': 0,
        'Marital_Status_encoded': 0,
        'Employment_Type_encoded': 0,
        'Loan_Purpose_encoded': 0
    }])[feature_cols]

    scaled = scaler.transform(sample_df)
    pred = model.predict(scaled)
    proba = model.predict_proba(scaled)

    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0][1] <= 1.0
    assert np.isclose(proba[0][0] + proba[0][1], 1.0)


def test_preprocessing_pipeline():
    """Verify preprocessing correctly encodes categorical values."""
    raw_df = pd.DataFrame([{
        'Age': 40,
        'Gender': 'Female',
        'Marital_Status': 'Married',
        'Employment_Type': 'Salaried',
        'Loan_Purpose': 'Home'
    }])

    encoded = encode_categorical_features(raw_df, CATEGORICAL_MAPPINGS)
    assert encoded['Gender_encoded'].iloc[0] == 1
    assert encoded['Marital_Status_encoded'].iloc[0] == 1
    assert encoded['Employment_Type_encoded'].iloc[0] == 0
    assert encoded['Loan_Purpose_encoded'].iloc[0] == 0
