# Logging Configuration Guide

## Overview

This project implements standard Python logging across all scripts. Logs are written to files instead of cluttering the terminal output, while keeping important user-facing messages visible on screen.

---

## Logging Setup

### Configuration File: `logging_config.py`

The centralized logging configuration provides:

- **File Logging**: Detailed logs with timestamps and log levels
- **Console Logging**: Important messages displayed to users
- **Organized Structure**: Logs saved to `logs/` directory with timestamp-based filenames

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems (saved to file only)
- **INFO**: Confirmation that things are working as expected (console + file)
- **WARNING**: Something unexpected or potentially problematic (console + file)
- **ERROR**: A serious problem with details (console + file)

---

## Log File Organization

Log files are automatically created in the `logs/` directory:

```
logs/
├── preprocessing_20260520_143045.log
├── model_selection_20260520_144522.log
├── model_training_and_tuning_20260520_145033.log
├── threshold_optimization_20260520_145644.log
├── final_evaluation_20260520_150022.log
└── prediction_pipeline_20260520_150512.log
```

### Filename Format
`{script_name}_{YYYYMMDD_HHMMSS}.log`

---

## Updated Scripts

All main scripts have been updated to use logging:

### 1. preprocessing.py
- Logs dataset loading and shape
- Logs feature selection details
- Logs data quality checks (missing values, duplicates)
- Logs preprocessing steps completion

**Log file**: `logs/preprocessing_*.log`

### 2. model_selection.py
- Logs dataset loading
- Logs SMOTE application
- Logs model training progress for each model
- Logs performance metrics for each model
- Logs model comparison and selection

**Log file**: `logs/model_selection_*.log`

### 3. model_training_and_tuning.py
- Logs SMOTE application
- Logs GridSearchCV execution
- Logs best hyperparameters found
- Logs final model evaluation metrics

**Log file**: `logs/model_training_and_tuning_*.log`

### 4. threshold_optimization.py
- Logs threshold search progress
- Logs all evaluated thresholds
- Logs threshold filtering and selection
- Logs final performance metrics

**Log file**: `logs/threshold_optimization_*.log`

### 5. final_evaluation.py
- Logs data loading
- Logs model loading and threshold loading
- Logs performance metrics
- Logs class distribution analysis
- Logs confusion matrix details
- Logs visualization generation

**Log file**: `logs/final_evaluation_*.log`

### 6. prediction_pipeline.py
- Logs preprocessing artifacts loading
- Logs model loading
- Logs prediction progress
- Logs results saving

**Log file**: `logs/prediction_pipeline_*.log`

---

## Using Logs

### View Recent Logs

```bash
# See latest preprocessing log
type logs\preprocessing_*.log | tail -50

# On Linux/Mac
tail -50 logs/preprocessing_*.log
```

### Search Logs for Errors

```bash
# Find all ERROR entries
findstr "ERROR" logs/*.log

# On Linux/Mac
grep "ERROR" logs/*.log
```

### View Specific Script Logs

```bash
# Most recent preprocessing logs
dir logs\preprocessing_*.log /O-D

# See all model training logs
type logs\model_training_*.log
```

---

## Log File Contents Example

```
2026-05-20 14:30:45 - preprocessing - INFO - Loading dataset from datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv
2026-05-20 14:30:46 - preprocessing - INFO - Dataset Loaded - Shape: (1470, 35)
2026-05-20 14:30:46 - preprocessing - DEBUG - First 5 Rows:
    EmployeeNumber  Age  ...
0             101   41  ...
1             102   49  ...
...
2026-05-20 14:30:46 - preprocessing - INFO - Columns after removing useless columns: [Age, MonthlyIncome, ...]
2026-05-20 14:30:47 - preprocessing - INFO - Train-Test Split: X_train Shape=(976, 15), X_test Shape=(244, 15)
2026-05-20 14:30:47 - preprocessing - INFO - Processed datasets saved to processed_data/
2026-05-20 14:30:47 - preprocessing - INFO - Scaler and encoders saved to preprocessing_artifacts/
2026-05-20 14:30:47 - preprocessing - INFO - PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY
```

---

## Terminal Output vs Logs

### What You See in Terminal (INFO level)
```
✓ Test Data Loaded: 244 samples, 15 features
✓ Model loaded successfully!
✓ Optimal Threshold: 0.4700
✓ Predictions generated for 244 test samples

======================================================================
MODEL PERFORMANCE METRICS
======================================================================

Accuracy  : 0.8456 (84.56%)
Precision : 0.7234 (72.34%)
Recall    : 0.6892 (68.92%)
F1 Score  : 0.7060
ROC-AUC   : 0.8901

======================================================================
FINAL EVALUATION COMPLETED SUCCESSFULLY!
======================================================================
```

### What You Get in Log Files (DEBUG level)

**Same as above, plus:**
- Detailed dataframe printouts
- Classification reports
- Confusion matrices
- Feature lists
- Hyperparameter details
- Probability statistics
- Data quality information

---

## Benefits of This Logging System

✅ **Cleaner Terminal**: No data dumps cluttering the console  
✅ **Persistent Records**: Full execution logs saved for audit trails  
✅ **Easy Debugging**: Search logs for errors without running scripts again  
✅ **Timestamped**: Each log file knows when it was created  
✅ **Organized**: Separate log file for each script  
✅ **Hierarchical**: DEBUG info in files, INFO level shown on screen  
✅ **Standardized**: Python's standard logging module  

---

## API Reference

### Using Logging in Python Scripts

```python
from logging_config import setup_logging

# Initialize logger at script start
logger = setup_logging('my_script_name')

# Log at different levels
logger.debug("Detailed diagnostic info")      # File only
logger.info("General information")             # File + console
logger.warning("Something unexpected")         # File + console
logger.error("Error occurred", exc_info=True) # File + console with traceback

# Example usage
logger.info(f"Processing {len(data)} records")
logger.debug(f"DataFrame shape: {data.shape}")
logger.error(f"Failed to load model: {error}")
```

### Log Format

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
2026-05-20 14:30:45 - preprocessing - INFO - Dataset Loaded
```

---

## Troubleshooting

### Issue: Can't find logs
**Solution**: Check the `logs/` folder in the project root directory

### Issue: Logs not showing in file
**Solution**: Ensure `logs/` directory exists and is writable

### Issue: Too many log files
**Solution**: Logs are timestamped - you can manually delete old ones. Consider keeping last 5-10 runs.

### Issue: Script runs but no log file created
**Solution**: Make sure `setup_logging('script_name')` is called at the script start

---

## Best Practices

1. **Start scripts with setup_logging()**
   ```python
   from logging_config import setup_logging
   logger = setup_logging('script_name')
   ```

2. **Use appropriate log levels**
   - DEBUG: Data shapes, intermediate values, detailed reports
   - INFO: Major steps, results, important messages
   - WARNING: Potential issues, unexpected conditions
   - ERROR: Failures with details

3. **Log before and after important operations**
   ```python
   logger.info("Loading model...")
   model = joblib.load("path/to/model.pkl")
   logger.info("Model loaded successfully")
   ```

4. **Include context in log messages**
   ```python
   # Bad
   logger.info("Data processed")
   
   # Good
   logger.info(f"Processed {len(data)} records in {elapsed_time:.2f}s")
   ```

---

## Summary

The logging system provides:
- **Cleaner terminal output** with only important messages
- **Comprehensive log files** for each script with timestamps
- **Organized structure** in the `logs/` directory
- **Standard Python logging** for easy troubleshooting

All scripts now use this centralized logging configuration automatically!

