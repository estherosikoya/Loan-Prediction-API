import pandas as pd
import joblib
import os

# Load the saved model, scaler, and training columns
def load_model_and_scaler(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The directory '{model_path}' does not exist. Please provide a valid path.")
    
    model = joblib.load(f"{model_path}/xgb_model.pkl")
    scaler = joblib.load(f"{model_path}/scaler.pkl")
    training_columns = joblib.load(f"{model_path}/training_columns.pkl")
    return model, scaler, training_columns

# Preprocess the input data
def preprocess_input(data, scaler, training_columns):
    # Ensure the input data has the same preprocessing as the training data
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    data['zip_code'] = data['zip_code'].astype(str)
    binary_cols = ['securities_account', 'cd_account', 'online', 'creditcard']
    data[binary_cols] = data[binary_cols].astype(bool)
    data['education'] = data['education'].astype('category')
    
    # Feature engineering
    if 'ccavg' in data.columns and data['ccavg'].dtype == 'object':
        data['ccavg'] = data['ccavg'].str.replace("/", ".").astype(float) * 12  # Convert to yearly spending
    
    # Define features
    features = ['age', 'experience', 'income', 'zip_code', 'family', 'ccavg', 
                'education', 'mortgage', 'securities_account', 'cd_account', 
                'online', 'creditcard']
    data = data[features]
    
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, columns=['education'])
    
    # Align the test data with the training columns
    data = data.reindex(columns=training_columns, fill_value=0)
    
    # Scale the features
    data_scaled = scaler.transform(data)
    return data_scaled

# Make predictions
def make_predictions(input_data, model, scaler, training_columns):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data, scaler, training_columns)
    
    # Make predictions
    predictions = model.predict(preprocessed_data)
    return predictions

if __name__ == "__main__":
    # File paths
    model_path = "model"  # Path to the saved model and scaler
    input_file = "/Users/utente/Downloads/ML_PRJ2/new_data.csv"  # Path to your input data file
    
    # Load the model, scaler, and training columns
    print("Loading model, scaler, and training columns...")
    model, scaler, training_columns = load_model_and_scaler(model_path)
    
    # Load the input data
    print("Loading input data...")
    input_data = pd.read_csv(input_file)
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(input_data, model, scaler, training_columns)
    
    # Output the predictions
    print("Predictions:")
    print(predictions)



   