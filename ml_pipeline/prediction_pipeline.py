# PRODUCTION PREDICTION PIPELINE

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import setup_logging

# Initialize logging
logger = setup_logging('prediction_pipeline')

print("=" * 70)
print("EMPLOYEE ATTRITION PREDICTION PIPELINE")
print("=" * 70)
print(f"Pipeline Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# 1. LOAD PREPROCESSING ARTIFACTS


logger.info("Loading preprocessing artifacts...")

scaler = joblib.load("models/scaler.pkl")
overtime_encoder = joblib.load("models/overtime_encoder.pkl")
attrition_encoder = joblib.load("models/attrition_encoder.pkl")

# Load selected features
with open("data/artifacts/selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

logger.info(f"✓ Scaler loaded")
logger.info(f"✓ Encoders loaded")
logger.info(f"✓ Selected features loaded ({len(selected_features)} features)")
print(f"✓ Scaler loaded")
print(f"✓ Encoders loaded")
print(f"✓ Selected features loaded ({len(selected_features)} features)")


# 2. LOAD TRAINED MODEL


logger.info("Loading trained model...")
model = joblib.load("models/trained_model.pkl")
logger.info("✓ Model loaded successfully!")
print("✓ Model loaded successfully!")


# 3. LOAD OPTIMAL THRESHOLD


logger.info("Loading optimal prediction threshold...")
with open("models/threshold.txt", "r") as f:
    optimal_threshold = float(f.read().strip())
logger.info(f"✓ Optimal Threshold: {optimal_threshold:.4f}")
print(f"✓ Optimal Threshold: {optimal_threshold:.4f}")


# 4. FUNCTION TO PREPROCESS NEW DATA


def preprocess_employee_data(df):
    """
    Preprocess new employee data using trained preprocessing objects.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with employee features (raw, unscaled)
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for model prediction
    """
    
    logger.debug(f"Preprocessing {len(df)} employee records")
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Select only required features
    required_features = selected_features.copy()
    
    # Check for missing features
    missing_features = [f for f in required_features if f not in df_processed.columns]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only required features
    df_processed = df_processed[required_features]
    
    # Encode OverTime column (if exists in data)
    if 'OverTime' in df_processed.columns:
        df_processed['OverTime'] = overtime_encoder.transform(df_processed['OverTime'])
    
    # Scale features using trained scaler
    df_scaled = scaler.transform(df_processed)
    
    # Convert back to DataFrame
    df_scaled = pd.DataFrame(
        df_scaled,
        columns=required_features
    )
    
    return df_scaled


# 5. FUNCTION TO MAKE PREDICTIONS


def predict_attrition(new_data):
    """
    Make attrition predictions for new employee data.
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        Raw employee data (unprocessed)
    
    Returns:
    --------
    dict or pd.DataFrame
        Predictions with probabilities and recommendations
    """
    
    # Preprocess data
    logger.debug("Preprocessing new employee data...")
    processed_data = preprocess_employee_data(new_data)
    
    # Get probability predictions
    logger.debug("Generating probability predictions...")
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Apply optimal threshold
    logger.debug(f"Applying threshold: {optimal_threshold}")
    predictions = (probabilities >= optimal_threshold).astype(int)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Employee_Index': range(len(new_data)),
        'Attrition_Probability': probabilities,
        'Predicted_Attrition': predictions,
        'Risk_Level': ['Low Risk' if p < 0.33 else ('Medium Risk' if p < 0.67 else 'High Risk') 
                       for p in probabilities],
        'Recommendation': ['Monitor' if pred == 0 else 'Intervene' for pred in predictions]
    })
    
    return results


# 6. INTERACTIVE PREDICTION MODE


def run_prediction_pipeline():
    """
    Main function to run the prediction pipeline.
    Allows user to choose between:
    - Predicting on new CSV file
    - Manual entry of single employee
    - Demo prediction
    """
    
    print("\n" + "=" * 70)
    print("PREDICTION MODE SELECTION")
    print("=" * 70)
    print("\nChoose prediction mode:")
    print("1. Load predictions from CSV file")
    print("2. Predict single employee (manual entry)")
    print("3. Run demo prediction")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Mode 1: Load from CSV
        csv_path = input("\nEnter CSV file path: ").strip()
        
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            print(f"❌ Error: File not found at {csv_path}")
            return
        
        try:
            logger.info(f"Loading predictions from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} employee records from {csv_path}")
            print(f"✓ Loaded {len(df)} employee records from {csv_path}")
            
            # Make predictions
            logger.info("Generating predictions...")
            print("\nGenerating predictions...")
            predictions = predict_attrition(df)
            
            # Add original data to results
            results_with_data = pd.concat([df.reset_index(drop=True), predictions], axis=1)
            
            # Save results
            output_path = f"results/metrics/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_with_data.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
            print(f"✓ Predictions saved to: {output_path}")
            
            # Display summary
            print("\n" + "=" * 70)
            print("PREDICTION SUMMARY")
            print("=" * 70)
            print(predictions)
            print(f"\nTotal Employees: {len(predictions)}")
            print(f"Predicted Attrition: {predictions['Predicted_Attrition'].sum()} ({predictions['Predicted_Attrition'].sum()/len(predictions)*100:.1f}%)")
            print(f"Predicted to Stay: {(predictions['Predicted_Attrition']==0).sum()} ({(predictions['Predicted_Attrition']==0).sum()/len(predictions)*100:.1f}%)")
            
            # Risk distribution
            print("\nRisk Level Distribution:")
            print(predictions['Risk_Level'].value_counts().to_string())
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            print(f"❌ Error processing file: {str(e)}")
    
    elif choice == "2":
        # Mode 2: Manual single entry
        print("\n" + "=" * 70)
        print("MANUAL EMPLOYEE PREDICTION")
        print("=" * 70)
        print("\nEnter employee data (or press Enter to use defaults):")
        
        employee_data = {}
        defaults = {
            'Age': 35,
            'MonthlyIncome': 5000,
            'OverTime': 'No',
            'DistanceFromHome': 10,
            'JobSatisfaction': 3,
            'EnvironmentSatisfaction': 3,
            'WorkLifeBalance': 3,
            'StockOptionLevel': 1,
            'YearsAtCompany': 5,
            'YearsSinceLastPromotion': 1,
            'YearsWithCurrManager': 3,
            'TotalWorkingYears': 10,
            'NumCompaniesWorked': 2,
            'JobInvolvement': 3,
            'PercentSalaryHike': 12
        }
        
        for feature in selected_features:
            default_val = defaults.get(feature, 0)
            user_input = input(f"  {feature} [{default_val}]: ").strip()
            if user_input == "":
                employee_data[feature] = default_val
            else:
                try:
                    employee_data[feature] = float(user_input) if feature != 'OverTime' else user_input
                except:
                    employee_data[feature] = user_input
        
        # Create DataFrame
        logger.info("Creating prediction for single employee...")
        df_single = pd.DataFrame([employee_data])
        
        # Make prediction
        logger.info("Generating prediction...")
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        
        try:
            prediction = predict_attrition(df_single)
            
            prob = prediction.iloc[0]['Attrition_Probability']
            pred = prediction.iloc[0]['Predicted_Attrition']
            risk = prediction.iloc[0]['Risk_Level']
            rec = prediction.iloc[0]['Recommendation']
            
            print(f"\nEmployee Profile:")
            for feature, value in employee_data.items():
                print(f"  {feature}: {value}")
            
            print(f"\n{'='*70}")
            print(f"Attrition Probability: {prob:.4f} ({prob*100:.2f}%)")
            print(f"Predicted Status: {'⚠️  WILL LEAVE' if pred == 1 else '✓ WILL STAY'}")
            print(f"Risk Level: {risk}")
            print(f"Recommendation: {rec}")
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
            print(f"{'='*70}")
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            print(f"❌ Error making prediction: {str(e)}")
    
    elif choice == "3":
        # Mode 3: Demo with sample data
        print("\n" + "=" * 70)
        print("DEMO PREDICTION")
        print("=" * 70)
        
        # Create sample employee data
        demo_data = pd.DataFrame({
            'Age': [25, 35, 45, 55],
            'MonthlyIncome': [2500, 5000, 7500, 10000],
            'OverTime': ['Yes', 'No', 'Yes', 'No'],
            'DistanceFromHome': [5, 10, 15, 20],
            'JobSatisfaction': [2, 3, 4, 4],
            'EnvironmentSatisfaction': [2, 3, 4, 4],
            'WorkLifeBalance': [2, 3, 3, 4],
            'StockOptionLevel': [0, 1, 2, 3],
            'YearsAtCompany': [1, 5, 10, 15],
            'YearsSinceLastPromotion': [0, 1, 2, 3],
            'YearsWithCurrManager': [0, 2, 5, 8],
            'TotalWorkingYears': [1, 10, 20, 30],
            'NumCompaniesWorked': [2, 2, 3, 4],
            'JobInvolvement': [2, 3, 4, 4],
            'PercentSalaryHike': [10, 12, 15, 15]
        })
        
        logger.info("Running demo prediction with sample employees...")
        print("\nGenerating predictions for 4 demo employees...")
        predictions = predict_attrition(demo_data)
        
        # Combine with original data
        results_demo = pd.concat([demo_data.reset_index(drop=True), predictions], axis=1)
        
        print("\n" + "=" * 70)
        print("DEMO RESULTS")
        print("=" * 70)
        print(results_demo.to_string(index=False))
        
        # Save demo results
        demo_path = f"results/metrics/demo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_demo.to_csv(demo_path, index=False)
        logger.info(f"Demo results saved to: {demo_path}")
        print(f"\n✓ Demo results saved to: {demo_path}")
    
    else:
        logger.warning("Invalid choice entered")
        print("❌ Invalid choice. Please enter 1, 2, or 3.")


# 7. EXAMPLE USAGE (uncomment to run automatically)


if __name__ == "__main__":
    
    # Option A: Run interactive mode
    run_prediction_pipeline()
    
    # Option B: Batch prediction example (uncomment to use)
    # print("\nRunning batch prediction example...")
    # df_new = pd.read_csv("path/to/new_employees.csv")
    # predictions = predict_attrition(df_new)
    # print(predictions)


print("\n" + "=" * 70)
print("PREDICTION PIPELINE COMPLETED")
print("=" * 70)
