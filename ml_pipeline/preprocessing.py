
# PREPROCESSING PIPELINE

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging('preprocessing')


# 2. LOAD DATASET


logger.info("Loading dataset from datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

logger.info(f"Dataset Loaded - Shape: {df.shape}")
logger.debug(f"First 5 Rows:\n{df.head()}")


# 3. REMOVE USELESS COLUMNS

remove_cols = [
    'EmployeeNumber',
    'EmployeeCount',
    'StandardHours',
    'Over18'
]

df.drop(columns=remove_cols, inplace=True)

logger.info(f"Columns after removing useless columns: {list(df.columns)}")


# 4. FEATURE SELECTION

# Selected based on:

selected_features = [
    'Age',
    'MonthlyIncome',
    'OverTime',
    'DistanceFromHome',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'StockOptionLevel',
    'YearsAtCompany',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'TotalWorkingYears',
    'NumCompaniesWorked',
    'JobInvolvement',
    'PercentSalaryHike'
]

target = 'Attrition'

# Keep only selected features + target
df = df[selected_features + [target]]

logger.info(f"Final Selected Features ({len(selected_features)}): {selected_features}")


# 5. CHECK MISSING VALUES
logger.info("Checking for missing values...")
logger.debug(f"Missing Values:\n{df.isnull().sum()}")


# 6. CHECK DUPLICATES
duplicate_count = df.duplicated().sum()
logger.info(f"Duplicate Rows: {duplicate_count}")

# Remove duplicates if any exist
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    logger.info("Duplicates removed.")


# 7. CHECK CLASS DISTRIBUTION
logger.info("Class Distribution Analysis")
class_dist = df['Attrition'].value_counts()
logger.debug(f"Class Distribution:\n{class_dist}")
pct_dist = df['Attrition'].value_counts(normalize=True) * 100
logger.debug(f"Percentage Distribution:\n{pct_dist}")


# 8. ENCODE CATEGORICAL COLUMNS
# Label Encoders
overtime_encoder = LabelEncoder()
attrition_encoder = LabelEncoder()

# Encode categorical columns
df['OverTime'] = overtime_encoder.fit_transform(df['OverTime'])
# No = 0, Yes = 1

df['Attrition'] = attrition_encoder.fit_transform(df['Attrition'])
# No = 0, Yes = 1


# 9. SPLIT FEATURES AND TARGET
X = df.drop('Attrition', axis=1)
y = df['Attrition']


# 10. TRAIN TEST SPLIT
# Stratify maintains class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

logger.info(f"Train-Test Split: X_train Shape={X_train.shape}, X_test Shape={X_test.shape}")


# 11. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

logger.info("Feature Scaling Completed")


# 12. SAVE PROCESSED DATASETS
df.to_csv(
    "data/processed/processed_attrition_dataset.csv",
    index=False
)
X_train_scaled.to_csv(
    "data/processed/X_train_scaled.csv",
    index=False
)
X_test_scaled.to_csv(
    "data/processed/X_test_scaled.csv",
    index=False
)
y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)
y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

logger.info("Processed datasets saved to processed_data/")


# 13. SAVE PREPROCESSING OBJECTS
joblib.dump(
    scaler,
    "models/scaler.pkl"
)
joblib.dump(
    overtime_encoder,
    "models/overtime_encoder.pkl"
)
joblib.dump(
    attrition_encoder,
    "models/attrition_encoder.pkl"
)
logger.info("Scaler and encoders saved to preprocessing_artifacts/")


# 14. SAVE FEATURE LIST
with open("data/artifacts/selected_features.txt", "w") as f:
    for feature in selected_features:
        f.write(feature + "\n")

logger.info("Feature list saved to preprocessing_artifacts/")


# PREPROCESSING COMPLETE
print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")