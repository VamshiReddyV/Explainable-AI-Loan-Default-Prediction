"""
Feature engineering utilities for loan risk assessment.
Computes financial ratios, debt-to-income indicators, and interaction features.
"""
import pandas as pd
import numpy as np


def compute_financial_ratios(df):
    """
    Generate derived financial and credit health metrics:
    - Debt-to-Income ratio (DTI)
    - Loan-to-Annual Income ratio (LTI)
    - Credit-Risk Index
    - Installment-to-Income estimate
    """
    df = df.copy()

    # Loan to Income Ratio
    if 'Loan_Amount' in df.columns and 'Annual_Income' in df.columns:
        df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Annual_Income'].replace(0, 1))

    # Estimated Monthly Installment (approximate simple interest)
    if 'Loan_Amount' in df.columns and 'Loan_Term' in df.columns and 'Interest_Rate' in df.columns:
        total_repayable = df['Loan_Amount'] * (1 + (df['Interest_Rate'] / 100) * (df['Loan_Term'] / 12))
        df['Estimated_EMI'] = total_repayable / df['Loan_Term'].replace(0, 1)
        
        if 'Monthly_Income' in df.columns:
            df['EMI_to_Income_Ratio'] = df['Estimated_EMI'] / df['Monthly_Income'].replace(0, 1)

    # Debt Pressure Index (existing loans * loan amount / income)
    if 'Existing_Loans' in df.columns and 'Loan_Amount' in df.columns and 'Annual_Income' in df.columns:
        df['Debt_Pressure_Index'] = (df['Existing_Loans'] + 1) * (df['Loan_Amount'] / df['Annual_Income'].replace(0, 1))

    return df
