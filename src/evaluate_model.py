"""
Comprehensive model evaluation metrics and explainability score validation.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_model_performance(data_path=None):
    """
    Run evaluation on the test set and return performance metrics dictionary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'raw', 'loan_data.csv')

    models_dir = os.path.join(base_dir, 'models')
    model = joblib.load(os.path.join(models_dir, 'loan_default_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))

    from src.train_model import preprocess_data
    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    X = df[feature_columns]
    y = df['Loan_Default']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    cm = confusion_matrix(y, y_pred)
    # cm: [[TN, FP], [FN, TP]]

    metrics = {
        'accuracy': round(float(accuracy_score(y, y_pred)) * 100, 2),
        'precision': round(float(precision_score(y, y_pred)) * 100, 2),
        'recall': round(float(recall_score(y, y_pred)) * 100, 2),
        'f1_score': round(float(f1_score(y, y_pred)) * 100, 2),
        'roc_auc': round(float(roc_auc_score(y, y_proba)) * 100, 2),
        'confusion_matrix': {
            'true_negative': int(cm[0][0]),
            'false_positive': int(cm[0][1]),
            'false_negative': int(cm[1][0]),
            'true_positive': int(cm[1][1])
        }
    }
    return metrics


if __name__ == '__main__':
    results = evaluate_model_performance()
    print("Model Performance Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v}")
