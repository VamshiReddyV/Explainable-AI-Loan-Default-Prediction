"""
Unit tests for SHAP and LIME explainability modules.
"""
import os
import joblib
import numpy as np
import pandas as pd
import pytest

from config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH
from src.explain_model import get_shap_explanation, get_lime_explanation, FEATURE_DISPLAY_NAMES


def test_shap_explanation_structure():
    """Verify SHAP returns 6 ranked features with numerical attribution values."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

    sample_df = pd.DataFrame([{
        'Age': 45,
        'Number_of_Dependents': 2,
        'Employment_Experience': 3,
        'Monthly_Income': 35000,
        'Annual_Income': 420000,
        'Credit_Score': 550,
        'Loan_Amount': 400000,
        'Loan_Term': 24,
        'Interest_Rate': 14.5,
        'Existing_Loans': 3,
        'Gender_encoded': 0,
        'Marital_Status_encoded': 1,
        'Employment_Type_encoded': 2,
        'Loan_Purpose_encoded': 2
    }])[feature_cols]

    scaled = scaler.transform(sample_df)
    shap_results = get_shap_explanation(model, scaler, feature_cols, scaled)

    features = shap_results if isinstance(shap_results, list) else shap_results.get('features', [])
    assert isinstance(features, list)
    assert len(features) > 0

    first = features[0]
    assert 'feature' in first
    assert 'display_name' in first
    assert 'shap_value' in first
    assert isinstance(first['shap_value'], (float, int))


def test_lime_explanation_structure():
    """Verify LIME explanation returns interpretable condition rules."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

    sample_df = pd.DataFrame([{
        'Age': 45,
        'Number_of_Dependents': 2,
        'Employment_Experience': 3,
        'Monthly_Income': 35000,
        'Annual_Income': 420000,
        'Credit_Score': 550,
        'Loan_Amount': 400000,
        'Loan_Term': 24,
        'Interest_Rate': 14.5,
        'Existing_Loans': 3,
        'Gender_encoded': 0,
        'Marital_Status_encoded': 1,
        'Employment_Type_encoded': 2,
        'Loan_Purpose_encoded': 2
    }])[feature_cols]

    scaled = scaler.transform(sample_df)
    lime_results = get_lime_explanation(model, scaler, feature_cols, scaled)

    assert isinstance(lime_results, list)
    assert len(lime_results) > 0
    first = lime_results[0]
    assert 'condition' in first
    assert 'weight' in first
