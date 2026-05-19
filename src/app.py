from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model, scaler, and training columns
model = joblib.load("final_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Loan Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert JSON data to a DataFrame
    input_data = pd.DataFrame([data])

    # Preprocess the input data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=training_columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)

    # Return the prediction as JSON
    result = {"prediction": int(prediction[0])}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)