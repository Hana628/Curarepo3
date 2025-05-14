import os
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Path to the saved model
MODEL_PATH = "attached_assets/bp_xgb.pkl"

# Try to load the model
try:
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            bp_model = pickle.load(f)
        logger.info("Blood pressure model loaded successfully")
    else:
        logger.warning(f"Blood pressure model file not found at {MODEL_PATH}")
        bp_model = None
except Exception as e:
    logger.error(f"Error loading blood pressure model: {str(e)}")
    bp_model = None

def predict_blood_pressure(data):
    """
    Predict blood pressure risk based on user data.
    
    Args:
        data: A dictionary containing user health metrics
        
    Returns:
        A dictionary containing prediction results
    """
    if bp_model is None:
        # Fallback if model isn't available
        logger.warning("Using fallback for blood pressure prediction")
        return fallback_bp_prediction(data)
    
    try:
        # Extract and preprocess features
        features = preprocess_bp_features(data)
        
        # Make prediction
        prediction = bp_model.predict_proba([features])[0]
        risk_score = prediction[1]  # Assuming binary classification with risk being class 1
        
        # Determine risk category
        if risk_score < 0.2:
            risk_category = "Low"
            advice = "Your blood pressure risk appears to be low. Continue with healthy habits!"
        elif risk_score < 0.5:
            risk_category = "Moderate"
            advice = "You have a moderate risk of high blood pressure. Consider reducing sodium intake and increasing physical activity."
        else:
            risk_category = "High"
            advice = "You have a higher risk of high blood pressure. Please consult with a healthcare provider for personalized advice."
        
        return {
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "advice": advice,
            "systolic_estimate": calculate_systolic_estimate(data, risk_score),
            "diastolic_estimate": calculate_diastolic_estimate(data, risk_score)
        }
    except Exception as e:
        logger.error(f"Error in blood pressure prediction: {str(e)}")
        return {
            "error": f"An error occurred during prediction: {str(e)}",
            "risk_category": "Unknown",
            "advice": "We couldn't process your data. Please try again or consult a healthcare provider."
        }

def preprocess_bp_features(data):
    """Extract and preprocess features for blood pressure prediction."""
    # Convert values to appropriate types
    age = float(data['age'])
    weight = float(data['weight'])
    height = float(data['height']) / 100  # cm to meters
    bmi = weight / (height * height)
    gender = 1 if data['gender'].lower() == 'male' else 0
    smoking = 1 if data['smoking'].lower() == 'yes' else 0
    alcohol = 1 if data['alcohol'].lower() == 'yes' else 0
    
    # Convert exercise level to numeric
    exercise_mapping = {
        'none': 0,
        'light': 1,
        'moderate': 2,
        'intense': 3
    }
    exercise = exercise_mapping.get(data['exercise'].lower(), 1)
    
    # Create feature array matching the original repository requirements
    # Based on https://github.com/Prateek-Alur/-Blood-Pressure-Abnormality-Detection
    
    # Extract all required features
    features = [
        float(data.get('age', 0)),
        float(data.get('bmi', bmi)),  # use calculated BMI if not provided
        int(data.get('sex', gender)),  # default to gender if sex not specified
        float(data.get('level_of_hemoglobin', 14)),  # default values for missing fields
        float(data.get('genetic_pedigree_coefficient', 0.5)),
        int(data.get('pregnancy', 0)),
        int(data.get('smoking', smoking)),  # use smoking value from earlier
        int(data.get('physical_activity', exercise)),  # map exercise to physical activity
        float(data.get('salt_content_in_diet', 2.0)),
        float(data.get('alcohol_consumption_per_day', 0 if alcohol == 0 else 1)),
        float(data.get('level_of_stress', 3)),
        int(data.get('chronic_kidney_disease', 0)),
        int(data.get('adrenal_and_thyroid_disorders', 0))
    ]
    
    return features

def calculate_systolic_estimate(data, risk_score):
    """Calculate an estimated systolic blood pressure value."""
    base = 120  # Normal systolic BP
    
    # Factors that might increase systolic BP
    age_factor = (float(data['age']) - 30) * 0.5 if float(data['age']) > 30 else 0
    weight_factor = (float(data['weight']) - 70) * 0.3 if float(data['weight']) > 70 else 0
    smoking_factor = 10 if data['smoking'].lower() == 'yes' else 0
    alcohol_factor = 5 if data['alcohol'].lower() == 'yes' else 0
    
    # Exercise decreases BP
    exercise_mapping = {
        'none': 0,
        'light': -2,
        'moderate': -5,
        'intense': -8
    }
    exercise_factor = exercise_mapping.get(data['exercise'].lower(), 0)
    
    # Calculate estimated systolic BP
    estimated_systolic = base + age_factor + weight_factor + smoking_factor + alcohol_factor + exercise_factor
    
    # Adjust based on risk score
    risk_adjustment = risk_score * 20
    estimated_systolic += risk_adjustment
    
    return min(max(int(estimated_systolic), 90), 180)  # Clamping to reasonable range

def calculate_diastolic_estimate(data, risk_score):
    """Calculate an estimated diastolic blood pressure value."""
    base = 80  # Normal diastolic BP
    
    # Factors that might increase diastolic BP
    age_factor = (float(data['age']) - 30) * 0.2 if float(data['age']) > 30 else 0
    weight_factor = (float(data['weight']) - 70) * 0.15 if float(data['weight']) > 70 else 0
    smoking_factor = 5 if data['smoking'].lower() == 'yes' else 0
    alcohol_factor = 3 if data['alcohol'].lower() == 'yes' else 0
    
    # Exercise decreases BP
    exercise_mapping = {
        'none': 0,
        'light': -1,
        'moderate': -3,
        'intense': -5
    }
    exercise_factor = exercise_mapping.get(data['exercise'].lower(), 0)
    
    # Calculate estimated diastolic BP
    estimated_diastolic = base + age_factor + weight_factor + smoking_factor + alcohol_factor + exercise_factor
    
    # Adjust based on risk score
    risk_adjustment = risk_score * 15
    estimated_diastolic += risk_adjustment
    
    return min(max(int(estimated_diastolic), 60), 120)  # Clamping to reasonable range

def fallback_bp_prediction(data):
    """Fallback method when model isn't available."""
    # Basic heuristic approach
    risk_points = 0
    
    # Age risk
    age = float(data['age'])
    if age > 60:
        risk_points += 3
    elif age > 45:
        risk_points += 2
    elif age > 35:
        risk_points += 1
    
    # BMI risk
    weight = float(data['weight'])
    height = float(data['height']) / 100  # cm to meters
    bmi = weight / (height * height)
    
    if bmi > 30:
        risk_points += 3  # Obese
    elif bmi > 25:
        risk_points += 2  # Overweight
    
    # Lifestyle factors
    if data['smoking'].lower() == 'yes':
        risk_points += 2
    
    if data['alcohol'].lower() == 'yes':
        risk_points += 1
    
    if data['exercise'].lower() == 'none':
        risk_points += 2
    elif data['exercise'].lower() == 'light':
        risk_points += 1
    
    # Calculate risk score from points
    max_points = 11
    risk_score = risk_points / max_points
    
    # Determine risk category
    if risk_score < 0.3:
        risk_category = "Low"
        advice = "Your blood pressure risk appears to be low. Continue with healthy habits!"
    elif risk_score < 0.6:
        risk_category = "Moderate"
        advice = "You have a moderate risk of high blood pressure. Consider reducing sodium intake and increasing physical activity."
    else:
        risk_category = "High"
        advice = "You have a higher risk of high blood pressure. Please consult with a healthcare provider for personalized advice."
    
    return {
        "risk_score": float(risk_score),
        "risk_category": risk_category,
        "advice": advice,
        "systolic_estimate": calculate_systolic_estimate(data, risk_score),
        "diastolic_estimate": calculate_diastolic_estimate(data, risk_score),
        "note": "This is an estimate based on general risk factors (model unavailable)"
    }
