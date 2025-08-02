import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create Model directory if not exists
os.makedirs('Model', exist_ok=True)

# === Dummy training data ===
# Number of features = length of input_vector in app.py
# SUBSTANCE_MAP (10) + UNIT_MAP (2) + SOURCE_MAP (3) + 7 numeric features = 22 features total

# Generate random sample data (100 samples)
X_dummy = np.random.rand(100, 22)

# Generate random target variable (continuous)
y_dummy = np.random.rand(100) * 100  # Example target values

# Initialize scaler and model
scaler = StandardScaler()
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Scale the features
X_scaled = scaler.fit_transform(X_dummy)

# Train the model
model.fit(X_scaled, y_dummy)

# Save the scaler and model
joblib.dump(scaler, os.path.join('Model', 'scaler.pkl'))
joblib.dump(model, os.path.join('Model', 'LR_model.pkl'))

print("Model and scaler trained and saved successfully!")
