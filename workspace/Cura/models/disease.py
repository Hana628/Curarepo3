"""
Disease Prediction Model for CURA Health Assistant.

This module provides functionality to predict diseases based on symptoms
using a pre-trained machine learning model.
"""

import os
import logging
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logger = logging.getLogger(__name__)

# Path to the model file
MODEL_PATH = os.path.join("attached_assets", "disease_prediction_model.pkl")

# Disease-specialist mapping
SPECIALIST_MAPPING = {
    "Fungal infection": "dermatologist",
    "Allergy": "allergist",
    "GERD": "gastroenterologist",
    "Chronic cholestasis": "gastroenterologist",
    "Drug Reaction": "allergist",
    "Peptic ulcer disease": "gastroenterologist",
    "AIDS": "infectious disease specialist",
    "Diabetes": "endocrinologist",
    "Gastroenteritis": "gastroenterologist",
    "Bronchial Asthma": "pulmonologist",
    "Hypertension": "cardiologist",
    "Migraine": "neurologist",
    "Cervical spondylosis": "orthopedist",
    "Paralysis (brain hemorrhage)": "neurologist",
    "Jaundice": "hepatologist",
    "Malaria": "infectious disease specialist",
    "Chicken pox": "infectious disease specialist",
    "Dengue": "infectious disease specialist",
    "Typhoid": "infectious disease specialist",
    "Hepatitis A": "hepatologist",
    "Hepatitis B": "hepatologist",
    "Hepatitis C": "hepatologist",
    "Hepatitis D": "hepatologist",
    "Hepatitis E": "hepatologist",
    "Alcoholic hepatitis": "hepatologist",
    "Tuberculosis": "pulmonologist",
    "Common Cold": "general physician",
    "Pneumonia": "pulmonologist",
    "Dimorphic hemorrhoids(piles)": "proctologist",
    "Heart attack": "cardiologist",
    "Varicose veins": "vascular surgeon",
    "Hypothyroidism": "endocrinologist",
    "Hyperthyroidism": "endocrinologist",
    "Hypoglycemia": "endocrinologist",
    "Osteoarthritis": "orthopedist",
    "Arthritis": "rheumatologist",
    "Psoriasis": "dermatologist",
    "Urinary tract infection": "urologist",
    "Acne": "dermatologist",
    "Impetigo": "dermatologist"
}

# Disease descriptions for common conditions
DISEASE_DESCRIPTIONS = {
    "Fungal infection": "A fungal infection caused by fungi that can affect the skin, nails, hair, or mucous membranes.",
    "Allergy": "An abnormal immune response to substances (allergens) that are typically harmless.",
    "GERD": "Gastroesophageal reflux disease, a digestive disorder that affects the lower esophageal sphincter.",
    "Chronic cholestasis": "A condition where bile flow from the liver is reduced or blocked, causing buildup of bile in the liver.",
    "Drug Reaction": "An adverse reaction to a medication that can range from mild to severe.",
    "Peptic ulcer disease": "Open sores that develop on the inner lining of the stomach and upper small intestine.",
    "Diabetes": "A metabolic disorder characterized by high blood sugar levels over a prolonged period.",
    "Hypertension": "Also known as high blood pressure, a condition where blood pressure is persistently elevated.",
    "Migraine": "A neurological condition characterized by recurrent headaches, often with throbbing pain on one side of the head.",
    "Common Cold": "A viral infectious disease of the upper respiratory tract affecting the nose, throat, sinuses, and larynx.",
    "Pneumonia": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
    "Heart attack": "A serious medical emergency where blood flow to part of the heart muscle is blocked.",
    "Arthritis": "Inflammation of one or more joints, causing pain and stiffness that can worsen with age.",
    "Urinary tract infection": "An infection in any part of the urinary system, including kidneys, bladder, ureters, and urethra.",
    "Acne": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells."
}

# Symptom list - should match the symptoms used in training the model
ALL_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
    'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 
    'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 
    'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

def predict_disease(symptoms):
    """
    Predict the disease based on the provided symptoms.
    
    Args:
        symptoms: A list of symptoms experienced by the patient
        
    Returns:
        A dictionary containing the prediction results
    """
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Disease prediction model file not found at {MODEL_PATH}")
            return fallback_disease_prediction(symptoms)
        
        # Load the model
        try:
            # Use a custom unpickler to handle compatibility issues
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle NumPy compatibility issues
                    if module == 'numpy._core':
                        module = 'numpy.core'
                    return super().find_class(module, name)
                    
            with open(MODEL_PATH, 'rb') as model_file:
                model = CustomUnpickler(model_file).load()
            logger.info("Successfully loaded disease prediction model")
            
            # Prepare input data
            input_data = process_symptoms(symptoms)
            
            # Make prediction
            prediction = model.predict([input_data])
            probabilities = model.predict_proba([input_data])
            
            # Get the disease with highest probability
            disease = prediction[0]
            confidence = round(max(probabilities[0]) * 100, 2)
            
            # Get specialist and description for the disease
            specialist = SPECIALIST_MAPPING.get(disease, "general physician")
            description = DISEASE_DESCRIPTIONS.get(disease, f"A medical condition that requires attention from a healthcare professional.")
            
            # Generate care recommendations
            care_recommendations = generate_care_recommendations(disease, symptoms)
            
            # Prepare response
            result = {
                "disease": disease,
                "confidence": confidence,
                "specialist": specialist,
                "description": description,
                "care_recommendations": care_recommendations
            }
            
            return result
        except Exception as model_error:
            logger.error(f"Model loading or prediction error: {str(model_error)}")
            return fallback_disease_prediction(symptoms)
    except Exception as e:
        logger.error(f"Error in disease prediction: {str(e)}")
        return fallback_disease_prediction(symptoms)

def process_symptoms(symptoms):
    """
    Convert the list of symptoms to a binary vector for model input.
    
    Args:
        symptoms: List of symptoms selected by the user
        
    Returns:
        A binary vector with 1s for symptoms present and 0s for symptoms absent
    """
    # Initialize a vector of zeros
    input_vector = [0] * len(ALL_SYMPTOMS)
    
    # Set 1s for symptoms that are present
    for symptom in symptoms:
        if symptom in ALL_SYMPTOMS:
            idx = ALL_SYMPTOMS.index(symptom)
            input_vector[idx] = 1
    
    return input_vector

def generate_care_recommendations(disease, symptoms):
    """
    Generate care recommendations based on the predicted disease and symptoms.
    
    Args:
        disease: The predicted disease
        symptoms: List of symptoms selected by the user
        
    Returns:
        A list of care recommendations
    """
    general_recommendations = [
        "Consult with a healthcare professional for proper diagnosis and treatment.",
        "Get adequate rest to help your body recover.",
        "Stay hydrated by drinking plenty of water.",
        "Monitor your symptoms and seek immediate medical attention if they worsen."
    ]
    
    # Disease-specific recommendations
    disease_specific = []
    
    if disease == "Common Cold":
        disease_specific = [
            "Use over-the-counter medications to relieve symptoms like congestion and cough.",
            "Gargle with warm salt water to relieve sore throat.",
            "Use a humidifier to ease congestion and coughing."
        ]
    elif disease == "Fungal infection":
        disease_specific = [
            "Keep affected areas clean and dry.",
            "Avoid sharing personal items that may spread the infection.",
            "Use antifungal creams as prescribed by your healthcare provider."
        ]
    elif disease == "Allergy":
        disease_specific = [
            "Identify and avoid potential allergens.",
            "Use antihistamines or allergy medications as advised by a doctor.",
            "Keep windows closed during high pollen seasons if you have seasonal allergies."
        ]
    elif disease == "Diabetes":
        disease_specific = [
            "Monitor your blood sugar levels regularly.",
            "Follow a balanced diet low in simple sugars.",
            "Take prescribed medications consistently.",
            "Exercise regularly as advised by your healthcare provider."
        ]
    elif disease == "Hypertension":
        disease_specific = [
            "Reduce sodium intake in your diet.",
            "Engage in regular physical activity.",
            "Take blood pressure medications as prescribed.",
            "Monitor your blood pressure regularly."
        ]
    elif "Hepatitis" in disease:
        disease_specific = [
            "Avoid alcohol and certain medications that may impact liver function.",
            "Follow a nutritious, well-balanced diet.",
            "Get plenty of rest as your body fights the infection."
        ]
    elif disease == "Migraine":
        disease_specific = [
            "Rest in a quiet, dark room during attacks.",
            "Apply cold or warm compresses to your head.",
            "Identify and avoid migraine triggers."
        ]
    elif disease == "Pneumonia":
        disease_specific = [
            "Complete the full course of prescribed antibiotics.",
            "Use a humidifier to loosen mucus.",
            "Practice deep breathing exercises to help clear your lungs."
        ]
    
    return general_recommendations + disease_specific

def fallback_disease_prediction(symptoms):
    """
    Provide a fallback prediction when the model isn't available.
    
    Args:
        symptoms: List of symptoms selected by the user
        
    Returns:
        A dictionary with fallback prediction results
    """
    logger.warning("Using fallback disease prediction")
    
    # Map certain symptom combinations to likely conditions
    if 'cough' in symptoms and 'high_fever' in symptoms and 'fatigue' in symptoms:
        disease = "Common Cold"
        specialist = "general physician"
        confidence = 75.0
    elif 'itching' in symptoms and 'skin_rash' in symptoms:
        disease = "Fungal infection"
        specialist = "dermatologist"
        confidence = 70.0
    elif 'headache' in symptoms and 'nausea' in symptoms:
        disease = "Migraine"
        specialist = "neurologist"
        confidence = 65.0
    else:
        disease = "Unknown condition"
        specialist = "general physician"
        confidence = 50.0
    
    # Get description for the disease
    description = DISEASE_DESCRIPTIONS.get(disease, "A condition that requires medical attention.")
    
    # Generate care recommendations
    care_recommendations = generate_care_recommendations(disease, symptoms)
    
    return {
        "disease": disease,
        "confidence": confidence,
        "specialist": specialist,
        "description": description,
        "care_recommendations": care_recommendations,
        "note": "This is a preliminary assessment. The prediction model is currently being integrated."
    }