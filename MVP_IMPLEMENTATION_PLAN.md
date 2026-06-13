# MVP Implementation Plan

## Project Context

This project is a BSc final year academic project: a machine learning-based web application for heart disease risk prediction. The current system already includes a trained Random Forest model, scaler, dataset, Jupyter notebooks, and a basic Flask web interface.

The MVP should demonstrate that the project is more than a notebook experiment. It should show a complete workflow: data preprocessing, trained model inference, user interaction, result interpretation, system persistence, and model evaluation.

## MVP Goal

Build a usable Flask web application that allows users to enter patient clinical information, receive a heart disease risk prediction, understand the main reasons behind the prediction, view model evaluation evidence, and keep a simple history of previous predictions.

## MVP Feature Set

### 1. Improved Prediction Form

**Purpose:** Provide a complete and user-friendly interface for entering patient information.

**Current status:** Basic form exists in `flask_app/templates/index.html`.

**Implementation tasks:**

- Keep all 13 model input features.
- Add realistic `min`, `max`, and `step` attributes to numeric fields.
- Preserve submitted form values after prediction or validation error.
- Add short helper text for unclear medical fields.
- Add a visible medical disclaimer near the result.

**Academic value:** Shows user-centered system design and responsible handling of medical prediction output.

### 2. Backend Input Validation

**Purpose:** Prevent invalid or unrealistic patient data from being passed into the model.

**Current status:** Backend directly converts form values to floats and catches broad exceptions.

**Implementation tasks:**

- Create a validation layer in `flask_app/app.py`.
- Validate required numeric fields:
  - `age`: 1-120
  - `trestbps`: 70-250
  - `chol`: 80-700
  - `thalch`: 60-250
  - `oldpeak`: 0-10
  - `ca`: 0-3
- Validate categorical values against allowed options.
- Return friendly error messages to the UI.
- Avoid exposing raw Python exception messages to users.

**Academic value:** Demonstrates software quality, safety, and robustness.

### 3. Prediction Result With Risk Interpretation

**Purpose:** Make the output understandable and useful.

**Current status:** App returns only high/low risk and probability.

**Implementation tasks:**

- Keep probability output.
- Add risk bands:
  - Low Risk: probability below 40%
  - Moderate Risk: probability from 40% to 69%
  - High Risk: probability 70% and above
- Display a short interpretation message for each band.
- Add a clear note that this is not a medical diagnosis.

**Academic value:** Makes the model result interpretable for non-technical users.

### 4. Simple Prediction Explanation

**Purpose:** Explain which patient inputs may have influenced the prediction.

**Current status:** No explanation exists.

**Implementation tasks:**

- Add a rule-based explanation section after prediction.
- Highlight notable risk-related inputs such as:
  - asymptomatic chest pain
  - exercise-induced angina
  - high oldpeak
  - higher number of major vessels
  - reversible thalassemia defect
  - low maximum heart rate
- Optionally add model feature importance later if time allows.

**Academic value:** Adds explainability without introducing heavy dependencies like SHAP during MVP.

### 5. Prediction History

**Purpose:** Turn the app from a one-off predictor into a small information system.

**Current status:** No persistence exists.

**Implementation tasks:**

- Add SQLite database support.
- Create a `predictions` table.
- Store:
  - timestamp
  - all patient input values
  - predicted risk band
  - predicted probability
- Add `/history` route.
- Display recent predictions in a table.
- Add a clear button only if needed later.

**Academic value:** Demonstrates database integration and practical system functionality.

### 6. Model Evaluation Page

**Purpose:** Show evidence that the model was trained and evaluated properly.

**Current status:** Evaluation exists mainly in the notebook and README.

**Implementation tasks:**

- Add `/model-info` route.
- Display:
  - dataset name
  - number of records: 920
  - model type: Random Forest Classifier
  - preprocessing summary
  - train/test split summary
  - performance metrics from the notebook/README
  - limitations of the model
- Add static text for confusion matrix/classification summary during MVP.
- Later, export plots from the notebook and display them as images.

**Academic value:** Helps during project defense because the model justification is visible inside the system.

### 7. JSON Prediction API

**Purpose:** Expose model inference as a backend service.

**Current status:** No API endpoint exists.

**Implementation tasks:**

- Add `/api/predict` POST endpoint.
- Accept JSON input with the same fields as the form.
- Reuse the same validation and prediction logic as the web form.
- Return JSON response:

```json
{
  "risk": "High Risk",
  "risk_band": "high",
  "probability": 0.82,
  "explanations": []
}
```

**Academic value:** Demonstrates separation of prediction logic from the web interface and supports future integrations.

### 8. Printable Prediction Report

**Purpose:** Allow users to save or print an assessment result.

**Current status:** No report feature exists.

**Implementation tasks:**

- Add a print-friendly result section.
- Add a browser print button after prediction.
- Include:
  - patient inputs
  - risk band
  - probability
  - timestamp
  - disclaimer

**Academic value:** Provides a practical output artifact and improves usability.

### 9. README Cleanup

**Purpose:** Make the repository accurate and professional.

**Current status:** README has encoding issues and inaccurate dataset sample count.

**Implementation tasks:**

- Fix broken characters.
- Update dataset size from 303 to 920.
- Update project structure.
- Document MVP features.
- Add setup/run instructions.
- Add academic disclaimer.

**Academic value:** Improves project presentation and reproducibility.

## Proposed Implementation Order

### Phase 1: Refactor Prediction Logic

- Create reusable validation function.
- Create reusable preprocessing/prediction function.
- Keep existing form behavior working.

### Phase 2: Improve UI and Result Output

- Add validation messages.
- Add risk bands.
- Add explanation section.
- Add disclaimer.
- Add print button.

### Phase 3: Add Persistence

- Add SQLite setup.
- Save predictions.
- Add `/history` page.

### Phase 4: Add Academic Support Pages

- Add `/model-info` page.
- Update navigation.
- Clean README.

### Phase 5: Add API Endpoint

- Add `/api/predict`.
- Reuse validation and prediction logic.
- Test with sample JSON payload.

## Suggested File Changes

```text
flask_app/
  app.py
  heart_predictions.db            # generated locally, ignored by git if desired
  templates/
    index.html
    history.html
    model_info.html
README.md
MVP_IMPLEMENTATION_PLAN.md
```

## Testing Plan

### Manual Web Tests

- Load home page.
- Submit valid low-risk input.
- Submit valid high-risk input.
- Submit invalid numeric values.
- Confirm friendly validation errors.
- Confirm prediction is saved to history.
- Confirm model info page loads.
- Confirm print button works.

### API Tests

- Send valid JSON to `/api/predict`.
- Send missing field JSON.
- Send invalid category JSON.
- Confirm JSON error response.

### Regression Checks

- Confirm feature column order still matches model training.
- Confirm scaler and model files load correctly.
- Confirm prediction probability is always between 0 and 1.

## MVP Completion Criteria

The MVP is complete when:

- Users can submit patient information from the web form.
- The app validates inputs before prediction.
- The app returns risk band, probability, and explanation.
- The app stores previous predictions.
- The history page displays saved predictions.
- The model info page explains dataset, model, metrics, and limitations.
- The API endpoint returns predictions in JSON.
- README is accurate and project-ready.

## Features To Defer Until After MVP

- User authentication.
- Doctor/admin roles.
- Cloud deployment.
- SHAP or LIME explainability.
- Full dashboard analytics.
- Mobile app.
- Email/SMS reports.
- Patient record management system.
- Multiple model selection from the UI.

## Defense Talking Points

- The system applies supervised machine learning to classify heart disease risk.
- Random Forest was selected after model comparison and tuning.
- The Flask web app demonstrates deployment of the trained model.
- Input validation improves reliability and prevents invalid inference.
- Prediction history demonstrates database integration.
- The model info page provides transparency about metrics and limitations.
- The project is educational and decision-support oriented, not a replacement for professional diagnosis.
