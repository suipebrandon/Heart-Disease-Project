from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Paths to the model and scaler
# They should be in the same folder as this script, or we can point back to the ML folder.
# For simplicity, we assume they're in the same directory.
MODEL_PATH = '../heart_disease_model.pkl'
SCALER_PATH = '../scaler.pkl'

# Load the trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    print("Error: heart_disease_model.pkl or scaler.pkl not found. Please run the notebook first!")
    exit(1)

# Define feature columns (to match training)
FEATURE_COLUMNS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'sex_Male',
                  'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 'fbs_True',
                  'restecg_normal', 'restecg_st-t abnormality', 'exang_True',
                  'slope_flat', 'slope_upsloping', 'thal_normal',
                  'thal_reversable defect']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_class = None
    
    if request.method == 'POST':
        try:
            # Extract basic numeric inputs
            data = {
                'age': float(request.form.get('age', 0)),
                'trestbps': float(request.form.get('trestbps', 0)),
                'chol': float(request.form.get('chol', 0)),
                'thalch': float(request.form.get('thalch', 0)),
                'oldpeak': float(request.form.get('oldpeak', 0)),
                'ca': float(request.form.get('ca', 0))
            }
            
            # Extract categorical inputs (One-Hot)
            data['sex_Male'] = 1 if request.form.get('sex') == 'Male' else 0
            
            cp = request.form.get('cp')
            data['cp_atypical angina'] = 1 if cp == 'atypical' else 0
            data['cp_non-anginal'] = 1 if cp == 'non-anginal' else 0
            data['cp_typical angina'] = 1 if cp == 'typical' else 0
            
            data['fbs_True'] = 1 if request.form.get('fbs') == 'True' else 0
            
            restecg = request.form.get('restecg')
            data['restecg_normal'] = 1 if restecg == 'normal' else 0
            data['restecg_st-t abnormality'] = 1 if restecg == 'st-t' else 0
            
            data['exang_True'] = 1 if request.form.get('exang') == 'True' else 0
            
            slope = request.form.get('slope')
            data['slope_flat'] = 1 if slope == 'flat' else 0
            data['slope_upsloping'] = 1 if slope == 'upsloping' else 0
            
            thal = request.form.get('thal')
            data['thal_normal'] = 1 if thal == 'normal' else 0
            data['thal_reversable defect'] = 1 if thal == 'reversable' else 0

            # Convert to DataFrame and reorder to match training
            patient_df = pd.DataFrame([data])
            patient_df = patient_df[FEATURE_COLUMNS]
            
            # Preprocess and Predict
            patient_scaled = scaler.transform(patient_df)
            prob = model.predict_proba(patient_scaled)[0][1]
            
            # Format output
            risk = "High Risk" if prob > 0.5 else "Low Risk"
            prediction_text = f"{risk} ({prob*100:.2f}% Probability)"
            prediction_class = "danger" if prob > 0.5 else "success"
            
        except Exception as e:
            prediction_text = f"Error processing input: {str(e)}"
            prediction_class = "warning"
            
    return render_template('index.html', prediction_text=prediction_text, prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
