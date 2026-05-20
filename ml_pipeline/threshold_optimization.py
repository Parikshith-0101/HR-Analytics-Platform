
# THRESHOLD SELECTION

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging('threshold_optimization')


# 2. LOAD TEST DATA
logger.info("Loading test data...")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv(
    "data/processed/y_test.csv"
).values.ravel()

logger.info(f"Test Data Loaded - X_test Shape: {X_test.shape}")


# 3. LOAD TRAINED MODEL
logger.info("Loading trained model...")
model = joblib.load(
    "models/trained_model.pkl"
)

logger.info("Model loaded successfully!")


logger.info("Generating probability predictions...")
y_probs = model.predict_proba(X_test)[:, 1]

logger.info("Starting threshold search...")

# BUSINESS LOGIC:
# We want:
# - High Recall
# - BUT acceptable Precision
#
# So:
# - DO NOT maximize Recall alone
# - Use F1 Score as primary selector
# - Maintain minimum Precision constraint

thresholds = np.arange(0.10, 0.95, 0.05)

results = []

logger.info(f"Threshold evaluation complete - tested {len(thresholds)} thresholds")
for threshold in thresholds:

    # Convert probabilities to predictions
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_probs)

    # Save results
    results.append({
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    })


results_df = pd.DataFrame(results)

logger.debug(f"All thresholds evaluated:\n{results_df}")


# 7. APPLY BUSINESS CONSTRAINT
logger.info("Applying precision constraint (>= 0.35)...")

filtered_df = results_df[
    results_df['Precision'] >= 0.35
]

logger.debug(f"Filtered thresholds (Precision >= 0.35):\n{filtered_df}")


# 8. SELECT BEST THRESHOLD
best_row = filtered_df.sort_values(
    by=['F1 Score', 'Recall'],
    ascending=False
).iloc[0]

best_threshold = best_row['Threshold']

logger.info(f"Best Threshold Selected: {best_threshold:.2f}")
logger.debug(f"Best Threshold Metrics: {best_row.to_dict()}")


# 9. FINAL PREDICTIONS


final_predictions = (
    y_probs >= best_threshold
).astype(int)


# 10. FINAL METRICS


final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions
)

final_recall = recall_score(
    y_test,
    final_predictions
)

final_f1 = f1_score(
    y_test,
    final_predictions
)

final_auc = roc_auc_score(
    y_test,
    y_probs
)

logger.info(f"Final Metrics - Accuracy: {final_accuracy:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}, ROC-AUC: {final_auc:.4f}")
logger.debug(f"Classification Report:\n{classification_report(y_test, final_predictions)}")


# 12. CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    final_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    f'Confusion Matrix (Threshold = {best_threshold:.2f})'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# 13. METRIC VS THRESHOLD PLOT


plt.figure(figsize=(10, 6))

plt.plot(
    results_df['Threshold'],
    results_df['Precision'],
    label='Precision'
)

plt.plot(
    results_df['Threshold'],
    results_df['Recall'],
    label='Recall'
)

plt.plot(
    results_df['Threshold'],
    results_df['F1 Score'],
    label='F1 Score'
)

plt.xlabel("Threshold")

plt.ylabel("Metric Score")

plt.title("Threshold vs Metrics")

plt.legend()

plt.grid(True)

plt.show()


# 14. PRECISION-RECALL CURVE


precisions, recalls, pr_thresholds = precision_recall_curve(
    y_test,
    y_probs
)

plt.figure(figsize=(8, 6))

plt.plot(recalls, precisions)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(True)

plt.show()


# 15. SAVE BEST THRESHOLD
logger.info("Saving best threshold...")
with open(
    "models/threshold.txt",
    "w"
) as f:

    f.write(str(best_threshold))

logger.info(f"Best threshold saved: {best_threshold}")


# 16. SAVE FINAL RESULTS
results_df.to_csv(
    "results/metrics/final_threshold_results.csv",
    index=False
)
logger.info("Threshold comparison results saved")

print("\n" + "=" * 60)
print("FINAL THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 60)
logger.info("FINAL THRESHOLD OPTIMIZATION COMPLETED")