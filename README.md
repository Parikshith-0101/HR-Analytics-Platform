# Employee Attrition Analytics Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

A professional, data-driven HR analytics platform that predicts employee voluntary attrition (turnover) risks and visualizes workforce demographics. Powered by a tuned, L1-regularized **Logistic Regression** model and wrapped in a responsive **Flask** full-stack web application.

---

## 🎯 Platform Features

* **Real-Time Flight Risk Assessment:** Interactively enter employee metrics to calculate an attrition probability score.
* **Prescriptive HR Interventions:** Dynamically yields specific "Primary Risk Factors" (e.g., overtime load, commute distance, low satisfaction) and tailored retention recommendations.
* **Macro Departmental Dashboards:** Responsive analytics dashboard utilizing **Chart.js** to summarize baseline turnover distributions, risk segments, and organizational drivers.
* **Calibrated Decision Boundary ($\theta = 0.65$):** Threshold optimized specifically under a strict Precision constraint ($\ge 35\%$) to maximize retention budget efficiency and eliminate "alarm fatigue."

---

## 🚀 Quick Start

### 1️⃣ Run the Web Application

Launch the Flask web server to access the HR platform locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask web server
python app.py
```

Open your browser and navigate to **`http://127.0.0.1:5000`** to access the Landing Page, Interactive Assessment Form, and Analytics Dashboard.

### 2️⃣ Run the Offline Machine Learning Pipeline

If you wish to retrain, tune, or evaluate the predictive models, the pipeline is modularly organized inside `ml_pipeline/`:

```bash
# Step 1: Preprocess the data and extract scaling weights
python ml_pipeline/preprocessing.py

# Step 2: Compare baseline models (Logistic Regression, Random Forest, AdaBoost)
python ml_pipeline/model_selection.py

# Step 3: Run SMOTE balancing and GridSearchCV hyperparameter tuning
python ml_pipeline/train_model.py

# Step 4: Calibrate the classification threshold to optimize F1 & Precision
python ml_pipeline/threshold_optimization.py

# Step 5: Generate performance charts, confusion matrices, and audit reports
python ml_pipeline/final_evaluation.py
```

---

## 📁 Project Structure

The codebase is organized according to clean separation of concerns and best-practice MLOps layouts:

```
├── data/
│   ├── raw/                  # Raw IBM Watson HR CSV files
│   ├── processed/            # Scaled, split, and preprocessed datasets
│   └── artifacts/            # Selected features metadata
│
├── ml_pipeline/              # Core machine learning pipeline scripts
│   ├── preprocessing.py      # Cleansing, label encoding, and standard scaling
│   ├── train_model.py        # GridSearchCV tuning and model serialization
│   ├── threshold_opt.py      # Custom decision boundary selection
│   └── final_evaluation.py   # Performance plots and audit report generator
│
├── models/                   # Serialized binaries (Pickles) & calibrated threshold
│   ├── trained_model.pkl     # Tuned Logistic Regression model
│   ├── scaler.pkl            # Pre-fit StandardScaler object
│   ├── overtime_encoder.pkl  # Label encoder for overtime
│   └── threshold.txt         # Plaintext optimal threshold (0.65)
│
├── results/                  # Quality control output metrics and diagnostics
│   ├── metrics/              # CSV threshold sweeps and parameter logs
│   └── plots/                # High-resolution performance charts (ROC/PR curves)
│
├── static/                   # Web assets (styles.css, fetch scripts, animations)
├── templates/                # Jinja2 layouts (base.html, index.html, predict.html)
│
├── app.py                    # Flask server entrypoint exposing predictive endpoints
└── requirements.txt          # Python library specifications
```

---

## 📊 Model Performance ($\theta = 0.65$)

Tested on a stratified test partition ($N=294$) with the following results:

| Metric | Score | Performance Details |
| :--- | :---: | :--- |
| **Accuracy** | **83.67%** | Correctness of all predictions on real un-sampled dataset |
| **Precision** | **48.94%** | Accuracy of flagged risks (1 in 2 flagged employees is a true flight risk) |
| **Recall** | **48.94%** | Coverage (captures nearly half of all actual exits inside the firm) |
| **F1-Score** | **0.4894** | Harmonic mean optimizing the precision-recall boundary |
| **ROC-AUC** | **75.48%** | Discrimination ability to rank positive cases higher than negative cases |

### 🎯 Test Confusion Matrix

```
                      Predicted Stay (0)   Predicted Leave (1)
                     +--------------------+--------------------+
   Actual Stay (0)   |      223 (TN)      |       24 (FP)      |
                     |  Correctly Stayed  |   False Alarm      |
                     +--------------------+--------------------+
   Actual Leave (1)  |       24 (FN)      |       23 (TP)      |
                     |   Missed Risk      |  Correctly Left    |
                     +--------------------+--------------------+
```

---

## 💡 Prediction Form Input Specification

The platform accepts evaluations based on **15 key HR attributes**:

1. **Age:** (18-60)
2. **MonthlyIncome:** ($1,000 - $20,000)
3. **OverTime:** ("Yes" / "No")
4. **DistanceFromHome:** (1 - 29 miles)
5. **JobSatisfaction:** (1-4: Low, Medium, High, Very High)
6. **EnvironmentSatisfaction:** (1-4: Low, Medium, High, Very High)
7. **WorkLifeBalance:** (1-4: Bad, Average, Good, Excellent)
8. **StockOptionLevel:** (0-3)
9. **YearsAtCompany:** (0 - 40 years)
10. **YearsSinceLastPromotion:** (0 - 15 years)
11. **YearsWithCurrManager:** (0 - 17 years)
12. **TotalWorkingYears:** (0 - 40 years)
13. **NumCompaniesWorked:** (0 - 9)
14. **JobInvolvement:** (1-4: Low, Medium, High, Very High)
15. **PercentSalaryHike:** (11% - 25%)

---

## 📖 In-Depth Project Documentation

For deeper academic and implementation explanations:
* **[ML_PROJECT_REPORT.md](ML_PROJECT_REPORT.md):** 8-Chapter comprehensive project documentation focusing on data science, SMOTE oversampling, GridSearchCV tuning, and threshold calibration.
* **[WEB_TECH_PROJECT_REPORT.md](WEB_TECH_PROJECT_REPORT.md):** 8-Chapter project documentation focusing on full-stack web architectures, Flask API routes, CSS3 responsive grid frameworks, asynchronous Fetch cycles, and Chart.js integrations.
* **[EXPLAINER.md](EXPLAINER.md):** A localized, intuitive ground-up guide explaining the platform's math, business rules, and design choices.

---

**Status:** ✅ Production-Ready  
**Contact:** Human Resources Analytics Platform Team  
**Last Updated:** May 20, 2026
