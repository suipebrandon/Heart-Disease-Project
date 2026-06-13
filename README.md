# Heart Disease Risk Prediction System

A BSc final year academic project that uses machine learning to estimate heart disease risk from patient clinical data. The project includes model training notebooks, saved model artifacts, and a Flask web application for prediction, interpretation, history tracking, and model information.

## Project Aim

The aim of this project is to develop a machine learning-based web application that can support heart disease risk assessment using selected clinical features from the UCI Heart Disease dataset.

This system is for educational and research purposes only. It is not a substitute for professional medical diagnosis or advice.

## MVP Features

- Patient risk assessment form using 13 clinical input features
- Backend validation for numeric ranges and categorical values
- Random Forest prediction with probability output
- Risk bands: Low Risk, Moderate Risk, and High Risk
- Rule-based explanation of important submitted risk indicators
- Local SQLite prediction history
- Model information page for academic defense and transparency
- JSON prediction API endpoint
- Print-friendly assessment report

## Dataset

- Source: UCI Heart Disease Dataset
- Local file: `heart_disease_uci.csv`
- Records in local CSV: 920
- Target: binary heart disease classification
- Original target column: `num`
- Binary conversion: `0` means no heart disease, values greater than `0` mean heart disease presence

The local dataset combines records from Cleveland, Hungary, Switzerland, and VA Long Beach.

## Project Structure

```text
.
├── Heart_Disease_Prediction.ipynb
├── ml.ipynb
├── heart_disease_uci.csv
├── heart_disease_model.pkl
├── scaler.pkl
├── requirements.txt
├── MVP_IMPLEMENTATION_PLAN.md
├── README.md
└── flask_app/
    ├── app.py
    ├── static/
    │   └── styles.css
    └── templates/
        ├── index.html
        ├── history.html
        └── model_info.html
```

## Technologies Used

- Python
- Flask
- pandas
- numpy
- scikit-learn
- joblib
- SQLite
- Bootstrap
- Jupyter Notebook

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Flask App

From the project root:

```bash
cd flask_app
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Application Pages

- `/` - patient assessment form and prediction result
- `/history` - recent saved predictions
- `/model-info` - dataset, model, metrics, and limitations
- `/api/predict` - JSON prediction endpoint

## API Example

Send a POST request to:

```text
/api/predict
```

Example JSON body:

```json
{
  "age": 55,
  "sex": "Male",
  "cp": "asymptomatic",
  "trestbps": 140,
  "chol": 250,
  "fbs": "False",
  "restecg": "normal",
  "thalch": 120,
  "exang": "True",
  "oldpeak": 2.1,
  "slope": "flat",
  "ca": 2,
  "thal": "reversable"
}
```

Example response:

```json
{
  "risk": "High Risk",
  "risk_band": "high",
  "probability": 0.82,
  "probability_percent": 82.0,
  "message": "The model predicts a high likelihood of heart disease.",
  "explanations": [],
  "timestamp": "2026-06-13 12:00:00"
}
```

## Model Summary

The main notebook compares multiple machine learning models, including:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest

The final saved model is a tuned Random Forest classifier. The README and notebook report approximately 85% accuracy and approximately 0.90 ROC-AUC.

## Academic Limitations

- The system is trained on a public dataset and has not been clinically validated.
- Some dataset fields contain missing values.
- The application provides decision-support information only.
- The result should always be interpreted by qualified medical professionals.
- Further work could include SHAP/LIME explainability, cloud deployment, authentication, and larger clinical datasets.

## Author

Final year BSc project.
