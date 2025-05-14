#!/usr/bin/env python
"""
Model verification script for CURA Health Assistant.
This script checks if all required model files are present and in the correct format.
It also verifies that the models can be loaded properly.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('verify_models')

def check_model_files():
    """Check if all required model files are present."""
    logger.info("Checking model files...")
    
    # Define the required model files
    required_files = [
        'attached_assets/bp_xgb.pkl',
        'attached_assets/diabetes_model.keras',
        'attached_assets/lifestyle_recommendation_model.pkl',
        'attached_assets/lstm_anomaly_model.keras'
    ]
    
    # Check each file
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
            logger.error(f"Missing model file: {file_path}")
    
    if missing_files:
        logger.error("Some model files are missing. Please download them from the GitHub release page.")
        return False
    
    logger.info("All model files are present.")
    return True

def verify_bp_model():
    """Verify that the blood pressure model can be loaded."""
    try:
        import joblib
        model_path = Path('attached_assets/bp_xgb.pkl')
        logger.info(f"Loading blood pressure model from {model_path}...")
        model = joblib.load(model_path)
        logger.info("Blood pressure model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load blood pressure model: {str(e)}")
        return False

def verify_diabetes_model():
    """Verify that the diabetes model can be loaded."""
    try:
        import tensorflow as tf
        model_path = Path('attached_assets/diabetes_model.keras')
        logger.info(f"Loading diabetes model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        logger.info("Diabetes model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load diabetes model: {str(e)}")
        return False

def verify_lifestyle_model():
    """Verify that the lifestyle recommendation model can be loaded."""
    try:
        import joblib
        model_path = Path('attached_assets/lifestyle_recommendation_model.pkl')
        logger.info(f"Loading lifestyle model from {model_path}...")
        model = joblib.load(model_path)
        logger.info("Lifestyle model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load lifestyle model: {str(e)}")
        return False

def verify_anomaly_model():
    """Verify that the anomaly detection model can be loaded."""
    try:
        import tensorflow as tf
        model_path = Path('attached_assets/lstm_anomaly_model.keras')
        logger.info(f"Loading anomaly detection model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        logger.info("Anomaly detection model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load anomaly detection model: {str(e)}")
        return False

def main():
    """Main verification function."""
    logger.info("Verifying CURA Health Assistant models...")
    
    # First, check if all model files are present
    if not check_model_files():
        return
    
    # Try to verify each model
    verify_results = []
    
    try:
        # Try loading dependencies
        import joblib
        import numpy as np
        
        # Verify all models
        verify_results.append(verify_bp_model())
        
        try:
            import tensorflow as tf
            verify_results.append(verify_diabetes_model())
            verify_results.append(verify_anomaly_model())
        except ImportError:
            logger.warning("TensorFlow not available, skipping diabetes and anomaly models verification")
        
        verify_results.append(verify_lifestyle_model())
        
    except ImportError as e:
        logger.error(f"Required dependency not available: {str(e)}")
        logger.error("Please install required dependencies: pip install -r requirements_github.txt")
        return
    
    # Check if all verifications passed
    if all(verify_results):
        logger.info("All models verified successfully!")
    else:
        logger.warning("Some models could not be verified. The application might still work with fallback mechanisms.")
    
    logger.info("")
    logger.info("Verification completed!")

if __name__ == '__main__':
    main()