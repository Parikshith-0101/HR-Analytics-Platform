
# MODEL TRAINING AND HYPERPARAMETER TUNING PIPELINE

import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging('model_training_and_tuning')


# 2. LOAD PREPROCESSED DATASETS
logger.info("Loading preprocessed datasets...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

logger.info(f"Datasets Loaded - X_train: {X_train.shape}, X_test: {X_test.shape}")


logger.info("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
logger.debug(f"SMOTE - Before: {pd.Series(y_train).value_counts().to_dict()}, After: {pd.Series(y_train_smote).value_counts().to_dict()}")


# 4. INITIALIZE BASE MODEL
base_model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=5000
)

# 5. DEFINE HYPERPARAMETER GRID
param_grid = {

    'C': [0.01, 0.1, 1, 10, 100],

    'penalty': ['l1', 'l2'],

    'solver': ['liblinear']
}

logger.info("Parameter Grid for GridSearchCV:")
logger.debug(f"Parameters: {param_grid}")


# 6. GRID SEARCH CV

# Optimize specifically for RECALL

logger.info("Running GridSearchCV...")

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(
    X_train_smote,
    y_train_smote
)


logger.info("GridSearchCV completed")
logger.debug(f"Best Parameters: {grid_search.best_params_}")
logger.debug(f"Best Cross-Validation Recall: {grid_search.best_score_}")


# 8. GET BEST MODEL
best_model = grid_search.best_estimator_


# 9. PREDICTIONS
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

logger.info(f"Tuned Model Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# 12. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix - Tuned Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 13. SAVE TUNED MODEL
joblib.dump(
    best_model,
    "models/trained_model.pkl"
)
logger.info("Tuned model saved to models/tuned_attrition_model.pkl")

# 14. SAVE BEST PARAMETERS
best_params_df = pd.DataFrame(
    [grid_search.best_params_]
)
best_params_df.to_csv(
    "results/metrics/best_hyperparameters.csv",
    index=False
)
logger.info("Best hyperparameters saved to results/best_hyperparameters.csv")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 60)
logger.info("HYPERPARAMETER TUNING COMPLETED")