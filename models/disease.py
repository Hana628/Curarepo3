"""
Disease Prediction Model for CURA Health Assistant.

This module provides functionality to predict diseases based on symptoms
using a pre-trained machine learning model.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logger = logging.getLogger(__name__)

# Path to the model file
MODEL_PATH = os.path.join("attached_assets", "disease_prediction_model.pkl")
# Make sure we are using fallback since model file is missing
model_file_exists = os.path.exists(MODEL_PATH)
logger.warning(f"Disease prediction model file exists: {model_file_exists}")

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
# This list is generated directly from the Symptom_Weights.csv file to ensure compatibility with the model
ALL_SYMPTOMS = [
    "abdominal_pain", "abnormal_menstruation", "acidity", "acute_liver_failure", "altered_sensorium",
    "anxiety", "back_pain", "belly_pain", "blackheads", "bladder_discomfort", "blister",
    "blood_in_sputum", "bloody_stool", "blurred_and_distorted_vision", "breathlessness",
    "brittle_nails", "bruising", "burning_micturition", "chest_pain", "chills",
    "cold_hands_and_feets", "coma", "congestion", "constipation", "continuous_feel_of_urine",
    "continuous_sneezing", "cough", "cramps", "dark_urine", "dehydration", "depression",
    "diarrhoea", "dischromic _patches", "distention_of_abdomen", "dizziness",
    "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", "extra_marital_contacts",
    "family_history", "fast_heart_rate", "fatigue", "fluid_overload", "foul_smell_of urine",
    "headache", "high_fever", "hip_joint_pain", "history_of_alcohol_consumption",
    "increased_appetite", "indigestion", "inflammatory_nails", "internal_itching",
    "irregular_sugar_level", "irritability", "irritation_in_anus", "joint_pain", "knee_pain",
    "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance", "loss_of_smell",
    "malaise", "mild_fever", "mood_swings", "movement_stiffness", "mucoid_sputum", "muscle_pain",
    "muscle_wasting", "muscle_weakness", "nausea", "neck_pain", "nodal_skin_eruptions",
    "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", "pain_in_anal_region",
    "painful_walking", "palpitations", "passage_of_gases", "patches_in_throat", "phlegm",
    "polyuria", "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "red_sore_around_nose",
    "red_spots_over_body", "redness_of_eyes", "restlessness", "runny_nose", "rusty_sputum",
    "scurring", "shivering", "silver_like_dusting", "sinus_pressure", "skin_peeling",
    "skin_rash", "slurred_speech", "small_dents_in_nails", "spinning_movements", "spotting_ urination",
    "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", "sweating",
    "swelled_lymph_nodes", "swelling_joints", "swelling_of_stomach", "swollen_blood_vessels",
    "swollen_extremeties", "swollen_legs", "throat_irritation", "toxic_look_(typhos)",
    "ulcers_on_tongue", "unsteadiness", "visual_disturbances", "vomiting", "watering_from_eyes",
    "weakness_in_limbs", "weakness_of_one_body_side", "weight_gain", "weight_loss",
    "yellow_crust_ooze", "yellow_urine", "yellowing_of_eyes", "yellowish_skin", "itching"
]

def predict_disease(symptoms):
    """
    Predict the disease based on the provided symptoms.
    
    Args:
        symptoms: A list of symptoms experienced by the patient
        
    Returns:
        A dictionary containing the prediction results
    """
    # If there's only one symptom and it's not very specific, use fallback (heuristic)
    if len(symptoms) == 1 and symptoms[0] in ['headache', 'fatigue', 'cough', 'nausea', 'fever', 'vomiting']:
        logger.warning(f"Single generic symptom detected: {symptoms[0]}, using fallback")
        return fallback_disease_prediction(symptoms)
        
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Disease prediction model file not found at {MODEL_PATH}")
            return fallback_disease_prediction(symptoms)
        
        # Load the doctor-disease recommendations for accurate specialist recommendations
        try:
            doctor_disease_path = os.path.join("attached_assets", "Doctor_Versus_Disease.csv")
            doctor_disease_df = pd.read_csv(doctor_disease_path, header=None, names=['disease', 'specialist'])
            # Create a mapping from disease to specialist
            doctor_disease_mapping = dict(zip(doctor_disease_df['disease'].str.strip(), 
                                             doctor_disease_df['specialist'].str.strip()))
            logger.info(f"Loaded doctor-disease mapping with {len(doctor_disease_mapping)} entries")
        except Exception as e:
            logger.warning(f"Could not load doctor-disease mapping: {str(e)}")
            doctor_disease_mapping = {}
            
        # Load disease descriptions if available
        try:
            disease_desc_path = os.path.join("attached_assets", "Disease_Description.csv")
            if os.path.exists(disease_desc_path):
                disease_desc_df = pd.read_csv(disease_desc_path)
                disease_descriptions = dict(zip(disease_desc_df['Disease'].str.strip(), 
                                                disease_desc_df['Description'].str.strip()))
                logger.info(f"Loaded disease descriptions with {len(disease_descriptions)} entries")
            else:
                disease_descriptions = {}
        except Exception as e:
            logger.warning(f"Could not load disease descriptions: {str(e)}")
            disease_descriptions = {}
        
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
                model_data = CustomUnpickler(model_file).load()
            
            # Handle both old and new model formats
            if isinstance(model_data, dict) and 'model' in model_data and 'features' in model_data:
                # New format: dict with model and feature names
                model = model_data['model']
                feature_names = model_data['features']
                logger.info(f"Successfully loaded disease prediction model with {len(feature_names)} features")
            else:
                # Assume it's just the model directly (old format)
                model = model_data
                feature_names = ALL_SYMPTOMS
                logger.info("Successfully loaded disease prediction model (old format)")
            
            # Create a dataframe with columns in the EXACT order used during training
            input_data = {}
            for feature in feature_names:
                input_data[feature] = [0]  # Initialize all features to 0
            input_df = pd.DataFrame(input_data)
            
            # Clean and normalize the input symptoms
            def clean_symptom(symptom):
                """Clean a symptom string by removing extra spaces and fixing common issues."""
                cleaned = symptom.strip().lower()
                # Fix specific formatting issues
                if "dischromic _patches" in cleaned:
                    cleaned = cleaned.replace("dischromic _patches", "dischromic_patches")
                if "foul_smell_of urine" in cleaned:
                    cleaned = cleaned.replace("foul_smell_of urine", "foul_smell_of_urine")
                if "spotting_ urination" in cleaned:
                    cleaned = cleaned.replace("spotting_ urination", "spotting_urination")
                return cleaned
            
            # Set provided symptoms to 1
            for symptom in symptoms:
                cleaned_symptom = clean_symptom(symptom)
                # Try exact match first
                if cleaned_symptom in feature_names:
                    input_df.loc[0, cleaned_symptom] = 1
                    continue
                
                # Try normalized match (replace underscore with space)
                for col in feature_names:
                    if cleaned_symptom.replace('_', ' ') == col.replace('_', ' '):
                        input_df.loc[0, col] = 1
                        break
            
            # Make prediction
            prediction = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
            
            # Get the disease with highest probability
            disease = prediction[0]
            confidence = round(max(probabilities[0]) * 100, 2)
            
            # If the model predicts a serious condition with only mild symptoms, use fallback
            serious_conditions = [
                "Paralysis (brain hemorrhage)", "Heart attack", "Tuberculosis", "AIDS"
            ]
            mild_symptoms = ['headache', 'fatigue', 'cough', 'nausea', 'fever']
            
            if disease in serious_conditions and all(s in mild_symptoms for s in symptoms):
                logger.warning(f"Detected unreliable prediction ({disease}) for mild symptoms, using fallback")
                return fallback_disease_prediction(symptoms)
            
            # Get specialist using the doctor-disease mapping if available
            if str(disease) in doctor_disease_mapping:
                specialist = doctor_disease_mapping[str(disease)]
                logger.info(f"Found specialist '{specialist}' for disease '{disease}' from Doctor_Versus_Disease.csv")
            else:
                specialist = SPECIALIST_MAPPING.get(str(disease), "general physician")
                logger.info(f"Using fallback specialist mapping for disease '{disease}'")
            
            # Get description from disease descriptions if available
            if str(disease) in disease_descriptions:
                description = disease_descriptions[str(disease)]
            else:
                description = DISEASE_DESCRIPTIONS.get(str(disease), 
                              f"A medical condition that requires attention from a healthcare professional.")
            
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
    
    # Enhanced symptom mapping to common conditions
    # Map certain symptom combinations to likely conditions
    if 'cough' in symptoms and 'high_fever' in symptoms and 'fatigue' in symptoms:
        disease = "Common Cold"
        specialist = "general physician"
        confidence = 75.0
    elif 'itching' in symptoms and 'skin_rash' in symptoms:
        disease = "Fungal infection"
        specialist = "dermatologist"
        confidence = 70.0
    elif 'headache' in symptoms and 'nausea' in symptoms and 'blurred_and_distorted_vision' in symptoms:
        disease = "Migraine"
        specialist = "neurologist"
        confidence = 72.0
    elif 'headache' in symptoms and 'nausea' in symptoms:
        disease = "Migraine"
        specialist = "neurologist"
        confidence = 65.0
    # Handle single symptoms
    elif len(symptoms) == 1 and 'headache' in symptoms:
        disease = "Tension Headache"
        specialist = "general physician"
        confidence = 60.0
    elif len(symptoms) == 1 and 'cough' in symptoms:
        disease = "Common Cold"
        specialist = "general physician"
        confidence = 55.0
    elif len(symptoms) == 1 and 'fatigue' in symptoms:
        disease = "Stress"
        specialist = "general physician"
        confidence = 50.0
    elif len(symptoms) == 1 and 'fever' in symptoms:
        disease = "Viral Infection"
        specialist = "general physician"
        confidence = 55.0
    elif len(symptoms) == 1 and 'nausea' in symptoms:
        disease = "Indigestion"
        specialist = "general physician"
        confidence = 60.0
    elif len(symptoms) == 1 and 'vomiting' in symptoms:
        disease = "Food Poisoning"
        specialist = "general physician"
        confidence = 60.0
    # Continue with the rest of the combinations
    elif 'chest_pain' in symptoms and 'breathlessness' in symptoms and 'fast_heart_rate' in symptoms:
        disease = "Heart attack"
        specialist = "cardiologist"
        confidence = 80.0
    elif 'yellowish_skin' in symptoms and 'yellowing_of_eyes' in symptoms:
        disease = "Jaundice"
        specialist = "hepatologist"
        confidence = 78.0
    elif 'stomach_pain' in symptoms and 'acidity' in symptoms:
        disease = "GERD"
        specialist = "gastroenterologist"
        confidence = 68.0
    elif 'acidity' in symptoms and 'ulcers_on_tongue' in symptoms:
        disease = "GERD"
        specialist = "gastroenterologist"
        confidence = 65.0
    elif 'continuous_sneezing' in symptoms and 'chills' in symptoms:
        disease = "Allergy"
        specialist = "allergist"
        confidence = 75.0
    elif 'shivering' in symptoms and 'chills' in symptoms and 'high_fever' in symptoms:
        disease = "Malaria"
        specialist = "infectious disease specialist"
        confidence = 77.0
    elif 'joint_pain' in symptoms and 'muscle_weakness' in symptoms:
        disease = "Arthritis"
        specialist = "rheumatologist"
        confidence = 69.0
    elif 'skin_rash' in symptoms and 'nodal_skin_eruptions' in symptoms:
        disease = "Chicken pox"
        specialist = "infectious disease specialist"
        confidence = 74.0
    elif 'burning_micturition' in symptoms and 'bladder_discomfort' in symptoms:
        disease = "Urinary tract infection"
        specialist = "urologist"
        confidence = 76.0
    elif 'fatigue' in symptoms and 'irregular_sugar_level' in symptoms:
        disease = "Diabetes"
        specialist = "endocrinologist"
        confidence = 73.0
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
        "note": "This is a preliminary assessment based on your symptoms."
    }