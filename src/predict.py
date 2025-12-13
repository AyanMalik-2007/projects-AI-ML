# src/predict.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and feature names
model = joblib.load('model/rf_model.pkl')
feature_names = joblib.load('model/feature_names.pkl')

def get_recommendation(prob_default):
    """Smart recovery recommendation based on Probability of Default"""
    if prob_default > 0.70:
        return "High Risk", "Immediate phone call to discuss repayment"
    elif prob_default > 0.40:
        return "Medium Risk", "Send SMS + follow-up email reminder"
    else:
        return "Low Risk", "Automated email reminder (monitor only)"

def predict_default(borrower_data):
    """
    Predict Probability of Default for a new borrower
    borrower_data: dict with same raw features as training data
    """
    # Convert to DataFrame
    df_input = pd.DataFrame([borrower_data])
    
    # Feature engineering (same as training)
    df_input['CreditAmount_per_Month'] = df_input['Credit amount'] / df_input['Duration']
    
    # One-hot encode categorical features
    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)
    
    # Align columns with training features (add missing, reorder)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]
    
    # Predict probability of default
    prob_default = model.predict_proba(df_encoded)[0][1]
    
    # Get recommendation
    risk_level, recommendation = get_recommendation(prob_default)
    
    return {
        'Probability_of_Default': round(prob_default, 3),
        'Risk_Level': risk_level,
        'Recommendation': recommendation
    }

# Example usage
if __name__ == "__main__":
    sample = {
        'Duration': 24,
        'Credit amount': 5000,
        'Sex': 'male',
        'Housing': 'own',
        'Saving accounts': 'little',
        'Checking account': 'moderate',
        'Purpose': 'car'
    }
    
    result = predict_default(sample)
    
    print("=== Prediction Result ===")
    print(f"Probability of Default: {result['Probability_of_Default']:.1%}")
    print(f"Risk Level: {result['Risk_Level']}")
    print(f"Recommendation: {result['Recommendation']}")