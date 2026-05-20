# LOGGING CONFIGURATION

import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


def setup_logging(script_name, log_level=logging.INFO):
    """
    Configure logging for a script with both file and console handlers.
    
    Parameters:
    -----------
    script_name : str
        Name of the script (used for log file naming)
    log_level : logging level
        Logging level (default: INFO)
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{LOGS_DIR}/{script_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(script_name):
    """
    Get an existing logger instance.
    
    Parameters:
    -----------
    script_name : str
        Name of the script
    
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(script_name)
