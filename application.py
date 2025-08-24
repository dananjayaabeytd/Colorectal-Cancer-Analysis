import joblib  
import numpy as np  
from flask import Flask, render_template, request  

# Initialize Flask application
app = Flask(__name__)

# Define paths to the trained model and scaler
model_path = "artifacts/models/model.pkl"
scaler_path = "artifacts/processed/scaler.pkl"

# Load the trained model and scaler from disk
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Home route: renders the main page with no predictions
@app.route('/')
def home():
    return render_template("index.html", predictions=None)

# Prediction route: handles form submission and returns prediction result
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract input features from the submitted form
        healthcare_cost = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        treatment_type = int(request.form["treatment_type"])
        diabetes = int(request.form["diabetes"])
        mortality_rate = float(request.form["mortality_rate"])

        # Prepare input for the model
        input = np.array([[healthcare_cost, tumor_size, treatment_type, diabetes, mortality_rate]])
        # Scale the input using the loaded scaler
        scaled_input = scaler.transform(input)
        # Make prediction using the loaded model
        prediction = model.predict(scaled_input)[0]

        # Render the result on the main page
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        # Return error message if something goes wrong
        return str(e)
    

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)