"""
Data preprocessing pipeline for loan default prediction dataset.
"""
import pandas as pd
import numpy as np
from config import CATEGORICAL_MAPPINGS, FEATURE_COLUMNS


def clean_dataset(df):
    """Clean missing values and remove anomalous outliers."""
    df = df.copy()
    # Fill numerical nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Enforce realistic bounds
    if 'Age' in df.columns:
        df['Age'] = df['Age'].clip(18, 100)
    if 'Credit_Score' in df.columns:
        df['Credit_Score'] = df['Credit_Score'].clip(300, 900)

    return df


def encode_categorical_features(df, mappings=None):
    """Map categorical columns to ordinal integer values."""
    if mappings is None:
        mappings = CATEGORICAL_MAPPINGS
    df = df.copy()
    for col, mapping in mappings.items():
        if col in df.columns:
            encoded_col = f'{col}_encoded'
            df[encoded_col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def prepare_features(df):
    """Preprocess, encode, and return feature matrix X and target y."""
    df_clean = clean_dataset(df)
    df_encoded = encode_categorical_features(df_clean)
    
    available_features = [c for c in FEATURE_COLUMNS if c in df_encoded.columns]
    X = df_encoded[available_features]
    y = df_encoded['Loan_Default'] if 'Loan_Default' in df_encoded.columns else None
    return X, y
