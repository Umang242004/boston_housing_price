import numpy as np
import pickle
from flask import Flask, request, render_template
import os
app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('linear_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure 'linear_regression_model.pkl' and 'scaler.pkl' are in the same directory as app.py")
    model = None
    scaler = None
    # You might want to exit or show an error page if essential files are missing

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html', prediction_text="Enter house features to predict price.")

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Model or scaler not loaded. Please check server logs.")

    # Get form data
    features = [
        float(request.form['crim']),
        float(request.form['zn']),
        float(request.form['indus']),
        float(request.form['chas']),
        float(request.form['nox']),
        float(request.form['rm']),
        float(request.form['age']),
        float(request.form['dis']),
        float(request.form['rad']),
        float(request.form['tax']),
        float(request.form['ptratio']),
        float(request.form['b']),
        float(request.form['lstat'])
    ]

    # Convert to numpy array and reshape for scaling
    final_features = np.array(features).reshape(1, -1)

    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(final_features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    # Format the prediction
    # Assuming price is in $1000s, convert to actual dollars and format
    predicted_price = f"${(prediction * 1000):,.2f}"

    return render_template('index.html', prediction_text=f"The predicted house price is: {predicted_price}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # default to 10000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
