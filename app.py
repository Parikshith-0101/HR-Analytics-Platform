import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load artifacts if they exist
SCALER_PATH = os.path.join("models", "scaler.pkl")
MODEL_PATH = os.path.join("models", "trained_model.pkl")
OVERTIME_ENCODER_PATH = os.path.join("models", "overtime_encoder.pkl")
FEATURES_PATH = os.path.join("data", "artifacts", "selected_features.txt")
THRESHOLD_PATH = os.path.join("models", "threshold.txt")

try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    overtime_encoder = joblib.load(OVERTIME_ENCODER_PATH)
    with open(FEATURES_PATH, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]
    with open(THRESHOLD_PATH, "r") as f:
        optimal_threshold = float(f.read().strip())
    print("Model artifacts loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model artifacts. Ensure preprocessing and training scripts have been run. Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Build DataFrame
        df = pd.DataFrame([data])
        
        # Ensure correct column order
        df = df[selected_features]
        
        # OverTime encoding
        if 'OverTime' in df.columns:
            df['OverTime'] = overtime_encoder.transform(df['OverTime'])
            
        # Scaling
        X_scaled_array = scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled_array, columns=selected_features)
        
        # Predict probability
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # Apply threshold
        prediction_val = int(prob >= optimal_threshold)
        status = "Likely To Leave" if prediction_val == 1 else "Likely To Stay"
        
        return jsonify({
            "prediction": status,
            "probability": float(prob),
            "threshold": float(optimal_threshold)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
