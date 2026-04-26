from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
saved = joblib.load("model/revenue_model.pkl")
model = saved["model"]
training_columns = saved["columns"]

@app.route("/")
def home():
    return "Revenue Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    new_transaction = pd.DataFrame([data])
    new_encoded = pd.get_dummies(new_transaction)

    for col in training_columns:
        if col not in new_encoded.columns:
            new_encoded[col] = 0

    new_encoded = new_encoded[training_columns]

    prediction = model.predict(new_encoded)[0]

    return jsonify({
        "predicted_revenue_tier": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)