"""
Generate synthetic loan data for training the default prediction model.
"""
import numpy as np
import pandas as pd
import os

def generate_sample_data(n_samples=5000, random_state=42):
    """Generate synthetic loan application data with realistic distributions."""
    np.random.seed(random_state)
    
    # --- Applicant Features ---
    ages = np.random.randint(21, 65, size=n_samples)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.55, 0.40, 0.05])
    
    marital_statuses = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples, p=[0.35, 0.55, 0.10])
    
    dependents = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.30, 0.25, 0.25, 0.12, 0.08])
    
    employment_types = np.random.choice(
        ['Salaried', 'Self-Employed', 'Freelancer', 'Unemployed'],
        size=n_samples, p=[0.45, 0.30, 0.15, 0.10]
    )
    
    employment_experience = np.zeros(n_samples)
    for i in range(n_samples):
        if employment_types[i] == 'Unemployed':
            employment_experience[i] = 0
        else:
            employment_experience[i] = np.random.randint(0, min(ages[i] - 18, 40) + 1)
    employment_experience = employment_experience.astype(int)
    
    # Monthly income depends on employment type and experience
    monthly_income = np.zeros(n_samples)
    for i in range(n_samples):
        base = {
            'Salaried': 40000, 'Self-Employed': 35000,
            'Freelancer': 25000, 'Unemployed': 5000
        }[employment_types[i]]
        monthly_income[i] = max(5000, base + employment_experience[i] * 2000 + np.random.normal(0, 15000))
    monthly_income = np.round(monthly_income, -2)  # Round to nearest 100
    
    annual_income = monthly_income * 12
    
    # --- Credit Features ---
    credit_scores = np.random.normal(650, 100, size=n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 300, 900)
    
    # --- Loan Features ---
    loan_amounts = np.random.choice(
        [50000, 100000, 150000, 200000, 250000, 300000, 400000, 500000, 750000, 1000000],
        size=n_samples
    )
    
    loan_purposes = np.random.choice(
        ['Home', 'Education', 'Personal', 'Business', 'Vehicle'],
        size=n_samples, p=[0.25, 0.15, 0.30, 0.15, 0.15]
    )
    
    loan_terms = np.random.choice([12, 24, 36, 48, 60, 120, 180, 240], size=n_samples)
    
    interest_rates = np.random.uniform(6.5, 18.0, size=n_samples).round(2)
    
    existing_loans = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    
    # --- Derived: Default Probability ---
    # Create a realistic target based on feature correlations
    default_score = np.zeros(n_samples, dtype=float)
    
    # Low credit score increases default risk
    default_score += (700 - credit_scores) / 200.0
    
    # High loan-to-income ratio increases risk
    lti_ratio = loan_amounts / (annual_income + 1)
    default_score += lti_ratio * 2.0
    
    # More existing loans = higher risk
    default_score += existing_loans * 0.3
    
    # Unemployment increases risk
    default_score += np.where(employment_types == 'Unemployed', 1.5, 0)
    default_score += np.where(employment_types == 'Freelancer', 0.3, 0)
    
    # Less experience increases risk
    default_score -= employment_experience * 0.05
    
    # Add noise
    default_score += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary with ~35% default rate
    threshold = np.percentile(default_score, 65)
    loan_default = (default_score >= threshold).astype(int)
    
    # --- Build DataFrame ---
    df = pd.DataFrame({
        'Full_Name': [f'Applicant_{i+1}' for i in range(n_samples)],
        'Age': ages,
        'Gender': genders,
        'Marital_Status': marital_statuses,
        'Number_of_Dependents': dependents,
        'Employment_Type': employment_types,
        'Employment_Experience': employment_experience,
        'Monthly_Income': monthly_income,
        'Annual_Income': annual_income,
        'Credit_Score': credit_scores,
        'Loan_Amount': loan_amounts,
        'Loan_Purpose': loan_purposes,
        'Loan_Term': loan_terms,
        'Interest_Rate': interest_rates,
        'Existing_Loans': existing_loans,
        'Loan_Default': loan_default
    })
    
    return df


if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'), exist_ok=True)
    
    df = generate_sample_data()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'loan_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples.")
    print(f"Default rate: {df['Loan_Default'].mean():.2%}")
    print(f"Saved to: {os.path.abspath(output_path)}")
    print(f"\nSample:\n{df.head()}")
