# src/train.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import os

# --- 1. Robust Path Configuration ---

# Get the path to the directory containing the current script (src/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Data path (one level up from src/, then into 'data/')
DATA_FILE_PATH = SCRIPT_DIR.parent / 'data' / 'german_credit_data.csv'

# Model directory path (one level up from src/, then into 'model/')
MODEL_DIR = SCRIPT_DIR.parent / 'model'
# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True) 

# --- 2. Load and Preprocess Function ---

def load_and_preprocess_data(path: Path):
    """Load and preprocess the German Credit dataset."""
    
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at: {path}. Please place 'german_credit_data.csv' in the 'data/' folder."
        )

    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    
    # Map the target variable: 'bad' = 1 (default), 'good' = 0 (no default)
    # The German Credit dataset typically uses 'Risk' for this column.
    df['Risk'] = df['Risk'].map({'bad': 1, 'good': 0})
    
    # Feature engineering: Credit per month
    df['CreditAmount_per_Month'] = df['Credit amount'] / df['Duration']
    
    # One-hot encode categorical columns
    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('Risk', axis=1)
    y = df_encoded['Risk']
    
    # Drop any non-numeric columns that might have slipped through (e.g., unnecessary IDs)
    X = X.select_dtypes(include=['number', 'bool'])
    
    return X, y, X.columns.tolist()

# --- 3. Training Function ---

def train_model():
    """Trains the Random Forest model and saves it."""
    try:
        X, y, feature_names = load_and_preprocess_data(path=DATA_FILE_PATH) 
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        return

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nStarting Model Training...")
    # Initialize and train Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        max_depth=8 # Added max_depth to prevent overfitting
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    print("\n=== Model Evaluation ===")
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save the trained model and feature names
    joblib.dump(model, MODEL_DIR / 'rf_model.pkl')
    joblib.dump(feature_names, MODEL_DIR / 'feature_names.pkl')
    print(f"\nModel and feature names saved successfully to {MODEL_DIR}!")

if __name__ == "__main__":
    train_model()