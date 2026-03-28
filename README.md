# Heart Disease Prediction ML 🫀

A machine learning application that predicts the risk of heart disease using patient data. Includes a Flask web interface for easy interaction.

## Features

- **Machine Learning Models**: Trained and compared multiple models including Logistic Regression, SVM, Decision Tree, and Random Forest
- **Best Model**: Random Forest Classifier with hyperparameter tuning via GridSearchCV
- **Web Interface**: Flask-based web application for making predictions
- **Data Analysis**: Comprehensive Exploratory Data Analysis (EDA) with visualizations
- **Model Evaluation**: ROC-AUC curves, confusion matrices, and classification reports

## Dataset

- **Source**: UCI Heart Disease Dataset
- **Samples**: 303 patient records
- **Features**: 13 clinical parameters (age, cholesterol, blood pressure, etc.)
- **Target**: Binary classification (presence or absence of heart disease)

## Project Structure

```
.
├── Heart_Disease_Prediction.ipynb     # Main ML notebook with model training
├── ml.ipynb                           # Additional ML experiments
├── heart_disease_uci.csv              # Dataset
├── heart_disease_model.pkl            # Trained Random Forest model
├── scaler.pkl                         # StandardScaler for preprocessing
├── flask_app/
│   ├── app.py                         # Flask application
│   └── templates/
│       └── index.html                 # Web interface
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Flask Web Application

```bash
cd flask_app
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`

### Option 2: Run the Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Heart_Disease_Prediction.ipynb` and run all cells

## Model Performance

- **Accuracy**: ~85%
- **ROC-AUC Score**: ~0.90
- **Cross-Validation Score**: Consistent across folds

## Input Features

The model accepts the following patient parameters:

| Feature | Type | Range |
|---------|------|-------|
| Age | Numeric | Years |
| Resting Blood Pressure | Numeric | mmHg |
| Serum Cholesterol | Numeric | mg/dl |
| Max Heart Rate | Numeric | bpm |
| ST Depression | Numeric | Continuous |
| Number of Vessels | Numeric | 0-4 |
| Sex | Categorical | Male/Female |
| Chest Pain Type | Categorical | 4 types |
| Fasting Blood Sugar | Categorical | True/False |
| Rest ECG | Categorical | 3 types |
| Exercise Induced Angina | Categorical | True/False |
| Slope | Categorical | 3 types |
| Thalassemia | Categorical | 3 types |

## Technologies Used

- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask
- **Model Serialization**: joblib

## Future Improvements

- Deploy on cloud platform (Heroku, AWS, Google Cloud)
- Add more data and retrain models
- Implement API endpoints
- Add user authentication
- Create mobile app version
- Implement model explainability (SHAP, LIME)

## License

This project is open source and available under the MIT License.

## Author

Your Name - [GitHub Profile](https://github.com/yourusername)

## Disclaimer

⚠️ **This application is for educational and research purposes only.** It should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical decisions.
