"""
Flask web application for Explainable AI Loan Default Prediction.
Provides interactive dashboard, loan prediction wizard, SHAP/LIME explainability studio,
and history tracking.
"""
import os
import io
import csv
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response

from src.database import (
    init_db, save_prediction, get_recent_predictions,
    get_all_predictions, get_dashboard_stats,
    get_risk_distribution, get_prediction_by_id
)
from src.explain_model import (
    get_shap_explanation, get_lime_explanation,
    FEATURE_DISPLAY_NAMES
)
from src.train_model import CATEGORICAL_MAPPINGS, FEATURE_COLUMNS


app = Flask(__name__)
app.config['SECRET_KEY'] = 'explainable-ai-loan-default-secret-key-2025'

# --- Load model artifacts at startup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model = joblib.load(os.path.join(MODELS_DIR, 'loan_default_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

# Initialize and seed database
init_db()


@app.route('/')
def dashboard():
    """Render the main executive dashboard matching Image 2."""
    stats = get_dashboard_stats()
    recent = get_recent_predictions(limit=5)
    risk_dist = get_risk_distribution()
    
    latest_prediction = recent[0] if recent else None
    latest_shap = latest_prediction.get('shap_explanation', []) if latest_prediction else []
    latest_lime = latest_prediction.get('lime_explanation', []) if latest_prediction else []
    
    return render_template('dashboard.html',
                           stats=stats,
                           recent_predictions=recent,
                           risk_distribution=risk_dist,
                           latest_shap=latest_shap,
                           latest_lime=latest_lime,
                           latest_prediction=latest_prediction,
                           now=datetime.now().strftime('%B %d, %Y'))


@app.route('/new-prediction')
def new_prediction():
    """Render the multi-step loan prediction wizard matching Image 1."""
    return render_template('new_prediction.html')


@app.route('/history')
@app.route('/applicants')
def history():
    """Render full prediction history and applicant directory."""
    status_filter = request.args.get('status', 'All')
    search = request.args.get('search', '')
    predictions = get_all_predictions(filter_status=status_filter, search_query=search)
    return render_template('history.html',
                           predictions=predictions,
                           current_status=status_filter,
                           current_search=search)


@app.route('/explainability')
@app.route('/shap')
@app.route('/lime')
def explainability():
    """Render the dedicated Explainability Studio with deep dives into SHAP & LIME."""
    recent = get_recent_predictions(limit=10)
    latest_prediction = recent[0] if recent else None
    return render_template('explainability.html',
                           latest_prediction=latest_prediction,
                           recent_predictions=recent)


@app.route('/model-performance')
@app.route('/risk-analysis')
def model_performance():
    """Render Model Performance metrics, confusion matrix, and ROC-AUC curve."""
    return render_template('performance.html')


@app.route('/about')
def about():
    """Render information about the platform, XAI methodology, and compliance."""
    return render_template('about.html')


@app.route('/prediction/<int:prediction_id>')
def view_prediction(prediction_id):
    """View a specific applicant's detailed explanation."""
    prediction = get_prediction_by_id(prediction_id)
    if prediction is None:
        return redirect(url_for('dashboard'))
    return render_template('prediction_result.html', prediction=prediction)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Receive form data, run inference, generate SHAP & LIME explanations, save to DB.
    Returns JSON with detailed explanation payloads.
    """
    try:
        data = request.get_json() or {}
        
        # Extract inputs with robust defaults
        applicant_name = data.get('full_name') or 'Unknown'
        age = int(data.get('age') or 30)
        gender = data.get('gender') or 'Male'
        marital_status = data.get('marital_status') or 'Single'
        dependents = int(data.get('number_of_dependents') or 0)
        employment_type = data.get('employment_type') or 'Salaried'
        employment_experience = int(data.get('employment_experience') or 0)
        monthly_income = float(data.get('monthly_income') or 0)
        annual_income = monthly_income * 12
        credit_score = int(data.get('credit_score') or 650)
        loan_amount = float(data.get('loan_amount') or 0)
        loan_purpose = data.get('loan_purpose') or 'Personal'
        loan_term = int(data.get('loan_term') or 12)
        interest_rate = float(data.get('interest_rate') or 10.0)
        existing_loans = int(data.get('existing_loans') or 0)
        
        # Categorical encoding
        gender_enc = CATEGORICAL_MAPPINGS['Gender'].get(gender, 0)
        marital_enc = CATEGORICAL_MAPPINGS['Marital_Status'].get(marital_status, 0)
        employment_enc = CATEGORICAL_MAPPINGS['Employment_Type'].get(employment_type, 0)
        purpose_enc = CATEGORICAL_MAPPINGS['Loan_Purpose'].get(loan_purpose, 2)
        
        # Construct DataFrame with column names to prevent scaler warnings
        features_dict = {
            'Age': [age],
            'Number_of_Dependents': [dependents],
            'Employment_Experience': [employment_experience],
            'Monthly_Income': [monthly_income],
            'Annual_Income': [annual_income],
            'Credit_Score': [credit_score],
            'Loan_Amount': [loan_amount],
            'Loan_Term': [loan_term],
            'Interest_Rate': [interest_rate],
            'Existing_Loans': [existing_loans],
            'Gender_encoded': [gender_enc],
            'Marital_Status_encoded': [marital_enc],
            'Employment_Type_encoded': [employment_enc],
            'Loan_Purpose_encoded': [purpose_enc]
        }
        df_input = pd.DataFrame(features_dict)[FEATURE_COLUMNS]
        
        # Scale features
        features_scaled = scaler.transform(df_input)
        
        # Predict class and probability
        prediction_class = int(model.predict(features_scaled)[0])
        prediction_proba = model.predict_proba(features_scaled)[0]
        default_probability = float(prediction_proba[1])
        prediction_label = 'Default' if prediction_class == 1 else 'Non-Default'
        
        # Risk tier determination
        if default_probability >= 0.70:
            risk_level = 'High Risk'
        elif default_probability >= 0.40:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'
        
        # Generate explanations
        raw_dict = {
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'existing_loans': existing_loans,
            'loan_term': loan_term,
            'monthly_income': monthly_income,
            'employment_type': employment_type
        }
        shap_explanation = get_shap_explanation(model, scaler, FEATURE_COLUMNS, features_scaled)
        lime_explanation = get_lime_explanation(model, scaler, FEATURE_COLUMNS, features_scaled, raw_dict=raw_dict)
        
        # Save record to database
        db_data = {
            'applicant_name': applicant_name,
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'number_of_dependents': dependents,
            'employment_type': employment_type,
            'employment_experience': employment_experience,
            'monthly_income': monthly_income,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'loan_term': loan_term,
            'interest_rate': interest_rate,
            'existing_loans': existing_loans,
            'prediction': prediction_label,
            'default_probability': default_probability,
            'risk_level': risk_level,
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation
        }
        prediction_id = save_prediction(db_data)
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'prediction': prediction_label,
            'default_probability': round(default_probability * 100, 2),
            'risk_level': risk_level,
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export-csv')
def export_csv():
    """Export all prediction records to a CSV file."""
    predictions = get_all_predictions()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'Applicant Name', 'Age', 'Gender', 'Employment Type',
        'Monthly Income', 'Credit Score', 'Loan Amount', 'Loan Term',
        'Interest Rate', 'Existing Loans', 'Prediction', 'Default Probability (%)',
        'Risk Level', 'Created At'
    ])
    
    for p in predictions:
        writer.writerow([
            p.get('id'),
            p.get('applicant_name'),
            p.get('age'),
            p.get('gender'),
            p.get('employment_type'),
            p.get('monthly_income'),
            p.get('credit_score'),
            p.get('loan_amount'),
            p.get('loan_term'),
            p.get('interest_rate'),
            p.get('existing_loans'),
            p.get('prediction'),
            round(p.get('default_probability', 0) * 100, 2),
            p.get('risk_level'),
            p.get('created_at')
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=loan_predictions.csv'}
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
