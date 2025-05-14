"""
Test script for the updated disease prediction model.
"""

import sys
import logging
from models.disease import predict_disease

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_common_symptoms():
    """Test disease prediction with common symptoms."""
    print("\n--- Testing common symptoms ---")
    
    test_cases = [
        ["headache"],
        ["nausea"],
        ["headache", "nausea"],
        ["cough", "fever"],
        ["cough", "high_fever", "fatigue"],
        ["skin_rash", "itching"],
        ["stomachache", "vomiting", "nausea"],
        ["joint_pain", "stiff_neck"],
        ["weakness_in_limbs", "high_fever"]
    ]
    
    for symptoms in test_cases:
        print(f"\nTesting symptoms: {', '.join(symptoms)}")
        result = predict_disease(symptoms)
        print(f"  Predicted disease: {result['disease']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Specialist: {result['specialist']}")

def test_kaggle_cases():
    """Test cases mentioned in the Kaggle notebook."""
    print("\n--- Testing Kaggle notebook cases ---")
    
    # Cases from the Kaggle notebook
    test_cases = [
        ["skin_rash", "nodal_skin_eruptions", "dischromic_patches"],  # Expected: Fungal infection
        ["continuous_sneezing", "shivering", "chills"],  # Expected: Common Cold
        ["joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain"],  # Expected: Dengue
        ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"] # Expected: Fungal infection
    ]
    
    for symptoms in test_cases:
        print(f"\nTesting symptoms: {', '.join(symptoms)}")
        result = predict_disease(symptoms)
        print(f"  Predicted disease: {result['disease']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Specialist: {result['specialist']}")

if __name__ == "__main__":
    print("Testing the updated disease prediction model")
    try:
        test_common_symptoms()
        test_kaggle_cases()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)