"""
Train a Random Forest model for loan default prediction.
Saves the model, scaler, and feature column list to the models/ directory.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Features used by the model (order matters)
FEATURE_COLUMNS = [
    'Age', 'Number_of_Dependents', 'Employment_Experience',
    'Monthly_Income', 'Annual_Income', 'Credit_Score',
    'Loan_Amount', 'Loan_Term', 'Interest_Rate', 'Existing_Loans',
    'Gender_encoded', 'Marital_Status_encoded',
    'Employment_Type_encoded', 'Loan_Purpose_encoded'
]

# Mappings for categorical features
CATEGORICAL_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'Marital_Status': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'Employment_Type': {'Salaried': 0, 'Self-Employed': 1, 'Freelancer': 2, 'Unemployed': 3},
    'Loan_Purpose': {'Home': 0, 'Education': 1, 'Personal': 2, 'Business': 3, 'Vehicle': 4}
}


def preprocess_data(df):
    """Encode categorical variables and prepare features."""
    df = df.copy()
    
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        encoded_col = f'{col}_encoded'
        df[encoded_col] = df[col].map(mapping).fillna(0).astype(int)
    
    return df


def train_model(data_path=None):
    """Load data, preprocess, train, evaluate, and save model artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'raw', 'loan_data.csv')
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess
    df = preprocess_data(df)
    
    X = df[FEATURE_COLUMNS]
    y = df['Loan_Default']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Non-Default', 'Default'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save artifacts
    joblib.dump(model, os.path.join(models_dir, 'loan_default_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(FEATURE_COLUMNS, os.path.join(models_dir, 'feature_columns.pkl'))
    
    # Also save categorical mappings for use in prediction
    joblib.dump(CATEGORICAL_MAPPINGS, os.path.join(models_dir, 'categorical_mappings.pkl'))
    
    print(f"\nModel artifacts saved to: {models_dir}")
    print(f"  - loan_default_model.pkl")
    print(f"  - scaler.pkl")
    print(f"  - feature_columns.pkl")
    print(f"  - categorical_mappings.pkl")
    
    return model, scaler, accuracy


if __name__ == '__main__':
    train_model()
