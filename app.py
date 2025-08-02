# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === Define Model Paths ===
model_path = os.path.join('Model', 'LR_model.pkl')
scaler_path = os.path.join('Model', 'scaler.pkl')

# === Attempt to Load Model and Scaler ===
model = None
scaler = None
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("WARNING: Model or Scaler file not found. Skipping prediction functionality.")

# === Categorical Mappings ===
SUBSTANCE_MAP = {
    'Carbon Dioxide': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Methane':        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'Nitrous Oxide':  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # Add more as needed...
}

UNIT_MAP = {
    'kg/2018 USD, purchaser price':          [1, 0],
    'kg CO2e/2018 USD, purchaser price':     [0, 1],
}

SOURCE_MAP = {
    'Commodity': [1, 0, 0],
    'Industry':  [0, 1, 0],
    'Transport': [0, 0, 1],
}

# === Home Page Route ===
@app.route('/')
def home():
    return render_template('index.html')

# === Predict Route ===
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template(
            'index.html',
            submitted=False,
            prediction_text="WARNING: Model is not loaded. Cannot perform prediction."
        )

    try:
        # === Get Form Inputs ===
        substance = request.form['substance']
        unit = request.form['unit']
        source = request.form['source']
        supply_wo_margin = float(request.form['supply_wo_margin'])
        margin = float(request.form['margin'])
        dq_reliability = float(request.form['dq_reliability'])
        dq_temporal = float(request.form['dq_temporal'])
        dq_geo = float(request.form['dq_geo'])
        dq_tech = float(request.form['dq_tech'])
        dq_data = float(request.form['dq_data'])

        # === Feature Vector ===
        input_vector = (
            SUBSTANCE_MAP[substance] +
            UNIT_MAP[unit] +
            SOURCE_MAP[source] +
            [
                supply_wo_margin,
                margin,
                dq_reliability,
                dq_temporal,
                dq_geo,
                dq_tech,
                dq_data
            ]
        )

        # === Predict ===
        input_array = np.array(input_vector).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        return render_template(
            'index.html',
            submitted=True,
            prediction_text=f"Estimated GHG Emission: {prediction:.4f} kg CO₂e/unit",
            inputs=[
                substance, unit, source,
                supply_wo_margin, margin,
                dq_reliability, dq_temporal,
                dq_geo, dq_tech, dq_data
            ]
        )

    except Exception as e:
        return render_template(
            'index.html',
            submitted=False,
            prediction_text=f"Error during prediction: {str(e)}"
        )

# === Run the App ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
