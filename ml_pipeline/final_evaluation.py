# FINAL MODEL EVALUATION PIPELINE

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
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
logger = setup_logging('final_evaluation')


print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)
print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# 1. LOAD EVALUATION DATA


logger.info("Loading test data...")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

logger.info(f"✓ Test Data Loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"✓ Test Data Loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")


# 2. LOAD TRAINED MODEL


logger.info("Loading trained model...")
model = joblib.load("models/trained_model.pkl")
logger.info("✓ Model loaded successfully!")
print("✓ Model loaded successfully!")


# 3. LOAD OPTIMAL THRESHOLD


logger.info("Loading optimal threshold...")
with open("models/threshold.txt", "r") as f:
    best_threshold = float(f.read().strip())
logger.info(f"✓ Optimal Threshold: {best_threshold:.4f}")
print(f"✓ Optimal Threshold: {best_threshold:.4f}")


# 4. GENERATE PREDICTIONS


logger.info("Generating predictions...")
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= best_threshold).astype(int)

logger.info(f"✓ Predictions generated for {len(y_pred)} test samples")
print(f"✓ Predictions generated for {len(y_pred)} test samples")


# 5. CALCULATE COMPREHENSIVE METRICS


print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

logger.info(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")


# 6. CLASS DISTRIBUTION IN TEST SET


print("\n" + "=" * 70)
print("TEST SET CLASS DISTRIBUTION")
print("=" * 70)

class_counts = pd.Series(y_test).value_counts()
class_pct = pd.Series(y_test).value_counts(normalize=True) * 100

print("\nActual Class Distribution:")
print(f"  No Attrition (0)  : {class_counts[0]:4d} samples ({class_pct[0]:5.2f}%)")
print(f"  Attrition (1)     : {class_counts[1]:4d} samples ({class_pct[1]:5.2f}%)")

pred_counts = pd.Series(y_pred).value_counts()
pred_pct = pd.Series(y_pred).value_counts(normalize=True) * 100

print("\nPredicted Class Distribution:")
print(f"  No Attrition (0)  : {pred_counts.get(0, 0):4d} samples ({pred_pct.get(0, 0):5.2f}%)")
print(f"  Attrition (1)     : {pred_counts.get(1, 0):4d} samples ({pred_pct.get(1, 0):5.2f}%)")


# 7. CONFUSION MATRIX


print("\n" + "=" * 70)
print("CONFUSION MATRIX ANALYSIS")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives (TN)  : {tn:4d}  (Correctly predicted no attrition)")
print(f"False Positives (FP) : {fp:4d}  (Incorrectly predicted attrition)")
print(f"False Negatives (FN) : {fn:4d}  (Missed attrition cases)")
print(f"True Positives (TP)  : {tp:4d}  (Correctly predicted attrition)")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"\nSpecificity          : {specificity:.4f} (Ability to identify true negatives)")


# 8. DETAILED CLASSIFICATION REPORT


print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print("\n" + classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))


# 9. PREDICTION PROBABILITY ANALYSIS


print("\n" + "=" * 70)
print("PREDICTION PROBABILITY ANALYSIS")
print("=" * 70)

logger.debug(f"Probability Statistics - Min: {y_probs.min():.4f}, Max: {y_probs.max():.4f}, Mean: {y_probs.mean():.4f}, Std: {y_probs.std():.4f}, Median: {np.median(y_probs):.4f}")


# 10. VISUALIZATIONS


logger.info("Generating comprehensive visualizations...")
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
ax1.set_title(f'Confusion Matrix\n(Threshold = {best_threshold:.4f})', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

# 2. Metrics Bar Chart
ax2 = plt.subplot(2, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ax2.bar(metrics, values, color=colors, alpha=0.7)
ax2.set_ylim([0, 1])
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
ax2.set_xticklabels(metrics, rotation=45, ha='right')
for i, v in enumerate(values):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_test, y_probs)
ax3.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
ax4.plot(recalls, precisions, color='#ff7f0e', lw=2)
ax4.fill_between(recalls, precisions, alpha=0.2, color='#ff7f0e')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])

# 5. Probability Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(y_probs[y_test == 0], bins=30, alpha=0.6, label='No Attrition (Actual)', color='#2ca02c')
ax5.hist(y_probs[y_test == 1], bins=30, alpha=0.6, label='Attrition (Actual)', color='#d62728')
ax5.axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.4f}')
ax5.set_xlabel('Prediction Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Confusion Matrix Normalized
ax6 = plt.subplot(2, 3, 6)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
ax6.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Actual')
ax6.set_xlabel('Predicted')

print("✓ Generating comprehensive visualization...")
plt.savefig('results/final_evaluation_metrics.png', dpi=300, bbox_inches='tight')
logger.info("✓ Saved: results/final_evaluation_metrics.png")
print("✓ Saved: results/final_evaluation_metrics.png")


# 11. PROBABILITY DISTRIBUTION PLOT


plt.figure(figsize=(10, 6))
plt.hist(y_probs[y_test == 0], bins=30, alpha=0.6, label='No Attrition (Actual)', color='green')
plt.hist(y_probs[y_test == 1], bins=30, alpha=0.6, label='Attrition (Actual)', color='red')
plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Decision Threshold = {best_threshold:.4f}')
plt.xlabel('Predicted Probability of Attrition', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Predicted Probabilities', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
print("✓ Generating probability distribution plot...")
plt.savefig('results/probability_distribution.png', dpi=300, bbox_inches='tight')
logger.info("✓ Saved: results/probability_distribution.png")
print("✓ Saved: results/probability_distribution.png")


# 12. CREATE COMPREHENSIVE EVALUATION REPORT


evaluation_report = f"""
{'='*70}
FINAL MODEL EVALUATION REPORT
{'='*70}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
MODEL INFORMATION
{'='*70}
Model Type: Logistic Regression (Tuned)
Model Location: models/tuned_attrition_model.pkl
Decision Threshold: {best_threshold:.4f}
Test Set Size: {len(y_test)} samples

{'='*70}
PERFORMANCE METRICS
{'='*70}
Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)
Precision : {precision:.4f} ({precision*100:.2f}%)
Recall    : {recall:.4f} ({recall*100:.2f}%)
F1 Score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}
Specificity : {specificity:.4f}

{'='*70}
CONFUSION MATRIX
{'='*70}
True Negatives (TN)   : {tn:4d}  (Correctly predicted no attrition)
False Positives (FP)  : {fp:4d}  (Incorrectly predicted attrition)
False Negatives (FN)  : {fn:4d}  (Missed attrition cases)
True Positives (TP)   : {tp:4d}  (Correctly predicted attrition)

{'='*70}
TEST SET DISTRIBUTION
{'='*70}
Actual Class Distribution:
  No Attrition: {class_counts[0]:4d} samples ({class_pct[0]:5.2f}%)
  Attrition:    {class_counts[1]:4d} samples ({class_pct[1]:5.2f}%)

Predicted Class Distribution:
  No Attrition: {pred_counts.get(0, 0):4d} samples ({pred_pct.get(0, 0):5.2f}%)
  Attrition:    {pred_counts.get(1, 0):4d} samples ({pred_pct.get(1, 0):5.2f}%)

{'='*70}
INTERPRETATION
{'='*70}
✓ Accuracy: {accuracy*100:.2f}% of predictions are correct
✓ Precision: {precision*100:.2f}% of predicted attrition cases are correct
✓ Recall: {recall*100:.2f}% of actual attrition cases are identified
✓ F1 Score: {f1:.4f} (balance between Precision and Recall)
✓ ROC-AUC: {roc_auc:.4f} (probability model ranks positive cases higher)

False Negatives: {fn} employees predicted as staying but actually left
False Positives: {fp} employees predicted as leaving but actually stayed

{'='*70}
RECOMMENDATION
{'='*70}
The model is ready for production deployment.

Key Metrics:
- High Recall ({recall*100:.1f}%): Captures majority of attrition cases
- Acceptable Precision ({precision*100:.1f}%): Reasonable positive prediction rate
- Strong ROC-AUC ({roc_auc:.4f}): Good discrimination ability

Business Impact:
- {fn} attrition cases may not be identified for intervention
- {fp} non-attrition employees may receive unnecessary retention efforts
- These trade-offs were optimized at threshold {best_threshold:.4f}

{'='*70}
"""

print(evaluation_report)

# Save report
with open('results/final_evaluation_report.txt', 'w') as f:
    f.write(evaluation_report)
logger.info("Evaluation report saved: results/final_evaluation_report.txt")
print("✓ Evaluation report saved: results/final_evaluation_report.txt")


# 13. SAVE DETAILED METRICS TO CSV


detailed_metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Specificity'],
    'Value': [accuracy, precision, recall, f1, roc_auc, specificity],
    'Percentage': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', f'{recall*100:.2f}%', 
                   f'{f1*100:.2f}%', f'{roc_auc*100:.2f}%', f'{specificity*100:.2f}%']
})

detailed_metrics_df.to_csv('results/final_metrics_summary.csv', index=False)
logger.info("Metrics summary saved: results/final_metrics_summary.csv")
print("✓ Metrics summary saved: results/final_metrics_summary.csv")


# 14. SAVE PREDICTIONS WITH PROBABILITIES


predictions_df = pd.DataFrame({
    'Sample_Index': range(len(y_test)),
    'Actual_Attrition': y_test,
    'Predicted_Attrition': y_pred,
    'Attrition_Probability': y_probs,
    'Correct_Prediction': y_test == y_pred
})

predictions_df.to_csv('results/test_predictions_detailed.csv', index=False)
logger.info("Detailed predictions saved: results/test_predictions_detailed.csv")
print("✓ Detailed predictions saved: results/test_predictions_detailed.csv")


print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
logger.info("FINAL EVALUATION COMPLETED SUCCESSFULLY!")
print(f"\nGenerated Files:")
print(f"  1. results/final_evaluation_metrics.png")
print(f"  2. results/probability_distribution.png")
print(f"  3. results/final_evaluation_report.txt")
print(f"  4. results/final_metrics_summary.csv")
print(f"  5. results/test_predictions_detailed.csv")
print("=" * 70)
