import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load and preprocess data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid path.")
    
    df = pd.read_csv(file_path)
    
    # Clean data
    df = df[df['Experience'] >= 0]
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['zip_code'] = df['zip_code'].astype(str)
    binary_cols = ['personal_loan', 'securities_account', 'cd_account', 'online', 'creditcard']
    df[binary_cols] = df[binary_cols].astype(bool)
    df['education'] = df['education'].astype('category')
    
    # Feature engineering
    if df['ccavg'].dtype == 'object':
        df['ccavg'] = df['ccavg'].str.replace("/", ".").astype(float) * 12  # Convert to yearly spending
    
    # Define features and target
    features = ['age', 'experience', 'income', 'zip_code', 'family', 'ccavg', 
                'education', 'mortgage', 'securities_account', 'cd_account', 
                'online', 'creditcard']
    X = df[features]
    y = df['personal_loan']
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, columns=['education'])
    
    return X, y

# Train and save the model
def train_and_save_model(X, y, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Handle class imbalance using SMOTETomek
    smote_tomek = SMOTETomek(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, 
                                                        test_size=0.4, random_state=42, stratify=y_resampled)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                    test_size=0.5, random_state=42, stratify=y_temp)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_val_pred = xgb_model.predict(X_val_scaled)
    logging.info("Validation Classification Report:")
    logging.info(classification_report(y_val, y_val_pred))
    
    # Save the model, scaler, and training columns
    joblib.dump(xgb_model, "final_xgb_model.pkl")  # Save the model with the correct name
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "training_columns.pkl")  # Save training columns
    logging.info("Model, scaler, and training columns saved successfully!")

if __name__ == "__main__":
    # File paths
    data_file = "/Users/utente/Downloads/Bank.csv"  # Replace with your dataset path
    model_output_path = "model"  # Replace with your desired model output directory
    
    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(data_file)
    
    # Train and save the model
    logging.info("Training and saving the model...")
    train_and_save_model(X, y, model_output_path)
    
    logging.info("Model training and saving completed successfully!")