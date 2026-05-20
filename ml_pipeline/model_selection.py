
# MODEL TRAINING PIPELINE
import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

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
logger = setup_logging('model_selection')


# 2. LOAD PREPROCESSED DATASETS
logger.info("Loading preprocessed datasets...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

logger.info(f"Datasets Loaded - X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")


# 3. HANDLE CLASS IMBALANCE USING SMOTE
logger.info("Applying SMOTE for class imbalance handling...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
logger.debug(f"SMOTE Applied - Before: {pd.Series(y_train).value_counts().to_dict()}, After: {pd.Series(y_train_smote).value_counts().to_dict()}")


# 4. INITIALIZE MODELS
models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    )
}


# 5. TRAIN + EVALUATE MODELS
results = []
trained_models = {}
for name, model in models.items():
    logger.info(f"Training model: {name}")

    # TRAIN MODEL
    model.fit(X_train_smote, y_train_smote)

    # Save trained model in dictionary
    trained_models[name] = model

    # PREDICTIONS
    y_pred = model.predict(X_test)

    # Probability predictions
    y_probs = model.predict_proba(X_test)[:, 1]


    # EVALUATION METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    logger.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    })

    logger.debug(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")


    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# 6. MODEL COMPARISON
results_df = pd.DataFrame(results)
logger.info("Model Comparison Complete")
logger.debug(f"Model Comparison:\n{results_df}")


# 7. SKIP VISUALIZATION - logged to file instead
logger.debug(f"Model Performance Metrics:\n{results_df}")


# 8. SELECT BEST MODEL
best_model_row = results_df.sort_values(
    by=['Recall', 'F1 Score', 'ROC-AUC'],
    ascending=False
).iloc[0]
best_model_name = best_model_row['Model']
best_model = trained_models[best_model_name]

logger.info(f"Best Model Selected: {best_model_name}")
logger.debug(f"Best Model Metrics: {best_model_row.to_dict()}")


# 9. SAVE BEST MODEL
joblib.dump(
    best_model,
    "models/best_attrition_model.pkl"
)
logger.info("Best model saved to models/best_attrition_model.pkl")

# 10. SAVE MODEL COMPARISON RESULTS
results_df.to_csv(
    "results/model_comparison_results.csv",
    index=False
)
logger.info("Model comparison results saved to results/model_comparison_results.csv")

print("\n" + "=" * 60)
print("MODEL TRAINING PIPELINE COMPLETED")
print("=" * 60)
logger.info("MODEL TRAINING PIPELINE COMPLETED")