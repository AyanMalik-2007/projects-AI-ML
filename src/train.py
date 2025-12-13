# src/train.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def load_and_preprocess_data(path='data/german_credit_data.csv'):
    """Load and preprocess the German Credit dataset"""
    df = pd.read_csv(path)
    
    # Target: 'bad' = 1 (default), 'good' = 0 (no default)
    df['Risk'] = df['Risk'].map({'bad': 1, 'good': 0})
    
    # Feature engineering
    df['CreditAmount_per_Month'] = df['Credit amount'] / df['Duration']
    
    # One-hot encode categorical columns
    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('Risk', axis=1)
    y = df_encoded['Risk']
    
    return X, y, X.columns  # Return feature names for later use

def train_model():
    X, y, feature_names = load_and_preprocess_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    print("=== Model Evaluation ===")
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save the trained model and feature names
    joblib.dump(model, 'model/rf_model.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    print("\nModel and feature names saved successfully!")

if __name__ == "__main__":
    train_model()