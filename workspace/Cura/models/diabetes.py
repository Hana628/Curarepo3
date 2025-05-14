import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "attached_assets/trained_model.h5"  # Correct the path to .h5 model
diabetes_model = None

try:
    import tensorflow as tf
    from models.custom_tf_loader import load_tf_model_safely
    logger.info("TensorFlow imported successfully")

    # Load model
    diabetes_model = load_tf_model_safely(MODEL_PATH)

    if diabetes_model is not None:
        diabetes_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Successfully loaded and compiled the diabetes model")
    else:
        raise ValueError("Model not loaded; please check the model file")

except ImportError as e:
    logger.error(f"TensorFlow import failed: {str(e)}")
    diabetes_model = None


def predict_diabetes(data):
    def calculate_risk_score(features):
        age, glucose, bmi, family_history, physical_activity = features
        score = 0.2 + age * 0.15 + glucose * 0.25 + bmi * 0.2 + family_history * 0.15 - physical_activity * 0.15
        return max(0, min(1, score))

    try:
        # Extract and normalize features
        age = float(data.get('age', 30))
        glucose = float(data.get('glucose', 100))
        bmi = float(data.get('bmi', 25))
        family_history = 1.0 if data.get('family_history', 'no').lower() == 'yes' else 0.0
        physical_activity = float(data.get('physical_activity', 0.5))

        # Normalizing features
        age_norm = (age - 18) / (85 - 18)
        glucose_norm = (glucose - 70) / 130
        bmi_norm = (bmi - 18.5) / 21.5
        physical_activity_norm = (physical_activity - 1) / 4

        features = np.array([age_norm, glucose_norm, bmi_norm, family_history, physical_activity_norm])

        # Ensure the input is reshaped as (1, 16, 1) because the model expects 16 features
        padded_features = np.zeros(16)
        padded_features[:5] = features
        X = padded_features.reshape(1, 16, 1)

        # Predict using the model
        if diabetes_model is not None:
            try:
                risk_score = float(diabetes_model.predict(X)[0][0])
                logger.info(f"Diabetes model prediction: {risk_score}")
            except Exception as model_error:
                logger.error(f"Model prediction failed: {str(model_error)}")
                risk_score = calculate_risk_score(features)
        else:
            risk_score = calculate_risk_score(features)

        # Categorize and generate recommendations based on the risk score
        if risk_score < 0.2:
            category = "Low"
            recommendations = ["Maintain a healthy lifestyle with regular physical activity."]
        elif risk_score < 0.5:
            category = "Moderate"
            recommendations = ["Increase physical activity to at least 150 minutes per week."]
        else:
            category = "High"
            recommendations = ["Consult with a healthcare provider about your risk."]

        return {
            "risk_score": round(risk_score, 2),
            "risk_category": category,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error predicting diabetes risk: {str(e)}")
        return {
            "error": "An error occurred during prediction.",
            "risk_score": 0.5,
            "risk_category": "Moderate",
            "recommendations": ["Eat a balanced diet.", "Exercise regularly."]
        }
