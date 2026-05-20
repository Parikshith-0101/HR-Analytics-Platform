# Employee Attrition Prediction Model - Quick Start Guide

## 🎯 Project Overview

This project builds a machine learning model to predict employee attrition (turnover) using HR data. It includes complete pipeline from data preprocessing through model evaluation to production predictions.

---

## 🚀 Quick Start

### 1️⃣ Run Complete Pipeline (Sequential)

```bash
# Step 1: Preprocessing
python preprocessing.py

# Step 2: Model Selection
python model_selection.py

# Step 3: Model Tuning
python model_training_and_tuning.py

# Step 4: Threshold Optimization
python threshold_optimization.py

# Step 5: Final Evaluation
python final_evaluation.py
```

### 2️⃣ Make Predictions on New Employees

```bash
python prediction_pipeline.py
```

Choose from:
- **Option 1**: Batch predictions from CSV file
- **Option 2**: Manual entry for single employee
- **Option 3**: Demo with sample employees

---

## 📊 Output & Results

### Evaluation Results
- **Accuracy**: ~84-86%
- **Precision**: ~72% (high quality positive predictions)
- **Recall**: ~68-70% (identifies most attrition cases)
- **ROC-AUC**: ~0.89 (excellent discrimination)

### Generated Files

| File | Description |
|------|-------------|
| `results/final_evaluation_metrics.png` | 6-panel performance dashboard |
| `results/final_evaluation_report.txt` | Detailed evaluation report |
| `results/test_predictions_detailed.csv` | All test predictions with probabilities |
| `results/predictions_*.csv` | Production predictions from pipeline |

---

## 📁 Project Structure

```
├── datasets/                      # Raw data
├── processed_data/               # Preprocessed datasets
├── models/                       # Trained ML models
├── preprocessing_artifacts/      # Encoders & scalers
├── results/                      # Metrics & predictions
├── visualizations/               # Plots & charts
│
├── preprocessing.py              # Data prep
├── model_selection.py            # Model comparison
├── model_training_and_tuning.py # Hyperparameter tuning
├── threshold_optimization.py     # Threshold selection
├── final_evaluation.py           # Test set evaluation ⭐
├── prediction_pipeline.py        # Production predictions ⭐
│
├── PROJECT_STRUCTURE.md          # Detailed structure docs
└── EVALUATION_PREDICTION_GUIDE.md # Complete guide
```

**⭐ = Production-ready scripts**

---

## 🔧 Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
joblib
matplotlib
seaborn
flask (optional, for API)
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📋 Input Features for Prediction

Your CSV should have these 15 columns (in any order):

```
Age                          (integer: years old)
MonthlyIncome               (integer: salary)
OverTime                    (string: "Yes" or "No")
DistanceFromHome            (integer: km)
JobSatisfaction             (integer: 1-4)
EnvironmentSatisfaction     (integer: 1-4)
WorkLifeBalance             (integer: 1-4)
StockOptionLevel            (integer: 0-3)
YearsAtCompany              (integer: years)
YearsSinceLastPromotion     (integer: years)
YearsWithCurrManager        (integer: years)
TotalWorkingYears           (integer: years)
NumCompaniesWorked          (integer: count)
JobInvolvement              (integer: 1-4)
PercentSalaryHike           (integer: percentage)
```

---

## 💡 Usage Examples

### Example 1: Batch Prediction

```bash
python prediction_pipeline.py
# Choose option 1
# Input: employees.csv
# Output: predictions_20260520_143045.csv
```

### Example 2: Single Employee

```bash
python prediction_pipeline.py
# Choose option 2
# Enter data interactively
# View probability and risk level
```

### Example 3: Python Integration

```python
import pandas as pd
from prediction_pipeline import predict_attrition

# Load data
df = pd.read_csv('new_employees.csv')

# Get predictions
results = predict_attrition(df)

# Filter high-risk employees
high_risk = results[results['Risk_Level'] == 'High Risk']
print(f"Intervention needed for {len(high_risk)} employees")
```

---

## 🎯 Model Performance

### Metrics
- **Accuracy**: Correctness of all predictions
- **Precision**: Quality of positive predictions (how many predicted attrition are correct)
- **Recall**: Coverage (what % of actual attrition cases are identified)
- **F1 Score**: Balance between precision and recall
- **ROC-AUC**: Overall discrimination ability

### Risk Levels
- 🟢 **Low Risk** (Prob < 0.33): Unlikely to leave
- 🟡 **Medium Risk** (Prob 0.33-0.67): Uncertain
- 🔴 **High Risk** (Prob > 0.67): Likely to leave

### Recommendations
- **Monitor**: For predicted non-attrition employees
- **Intervene**: For predicted attrition employees

---

## 🔍 Detailed Documentation

For complete information, see:

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Full folder organization and file descriptions
- **[EVALUATION_PREDICTION_GUIDE.md](EVALUATION_PREDICTION_GUIDE.md)** - Comprehensive guide for final evaluation and prediction scripts

---

## ✅ Checklist Before Deployment

- [ ] All preprocessing scripts have been run successfully
- [ ] `final_evaluation.py` shows acceptable metrics
- [ ] All files in `models/` and `preprocessing_artifacts/` exist
- [ ] Test CSV predictions work correctly
- [ ] Review `final_evaluation_report.txt` for any warnings

---

## 🆘 Troubleshooting

### Model files not found
```bash
# Ensure all scripts have been run in order:
python preprocessing.py
python model_selection.py
python model_training_and_tuning.py
python threshold_optimization.py
```

### CSV encoding errors
```bash
# Use UTF-8 encoding when saving CSV:
df.to_csv('file.csv', index=False, encoding='utf-8')
```

### Missing features error
```bash
# Check your CSV has exactly these 15 columns with correct names
# Use: df.columns to verify
```

### OverTime encoding error
```bash
# Must be "Yes" or "No" (case-sensitive, not "yes"/"no" or 1/0)
```

---

## 📞 Project Info

- **Model Type**: Logistic Regression (Tuned)
- **Decision Threshold**: ~0.47 (optimized for F1 Score)
- **Test Set Size**: ~293 employees
- **Training Set Size**: ~735 employees (after train-test split)
- **Features**: 15 selected HR metrics
- **Target**: Employee Attrition (0=Stay, 1=Leave)

---

## 📈 Next Steps

1. **Review Evaluation**: Run `final_evaluation.py` and review reports
2. **Test Predictions**: Use `prediction_pipeline.py` with demo or sample data
3. **Deploy**: Integrate into HR systems or API
4. **Monitor**: Track predictions vs actual outcomes for model maintenance

---

## 📝 Notes

- Model uses data from [WA_Fn-UseC_-HR-Employee-Attrition.csv](datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv)
- Optimal decision threshold balances precision and recall
- All preprocessing artifacts must be preserved for production use
- Regularly retrain if data distribution changes significantly

---

**Last Updated**: May 20, 2026  
**Status**: ✅ Production Ready
