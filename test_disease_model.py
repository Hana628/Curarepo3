"""
Test script for disease prediction model
"""
import os
import sys
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(".")

# Test directly loading and using the model
def test_direct_load():
    model_path = os.path.join("attached_assets", "disease_prediction_model.pkl")
    logger.info(f"Testing direct load of model from: {model_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    
    # Try direct loading
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Successfully loaded model directly with pickle")
        
        # Test prediction with direct model
        test_symptoms = ["headache", "nausea"]
        logger.info(f"Testing direct prediction with symptoms: {test_symptoms}")
        
        # Create input vector
        from models.disease import ALL_SYMPTOMS, process_symptoms
        input_vector = process_symptoms(test_symptoms)
        
        # Make prediction
        prediction = model.predict([input_vector])
        probabilities = model.predict_proba([input_vector])
        logger.info(f"Direct prediction result: {prediction[0]}")
        logger.info(f"Confidence: {max(probabilities[0]) * 100:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in direct load test: {str(e)}")

# Test using the module's predict_disease function
def test_module_function():
    logger.info("Testing disease prediction through module function")
    
    try:
        from models.disease import predict_disease
        
        # Test with headache
        test_symptoms = ["headache"]
        logger.info(f"Testing predict_disease with: {test_symptoms}")
        result = predict_disease(test_symptoms)
        logger.info(f"Function result: {result}")
        
        # Test with multiple symptoms
        test_symptoms = ["headache", "nausea", "vomiting"]
        logger.info(f"Testing predict_disease with: {test_symptoms}")
        result = predict_disease(test_symptoms)
        logger.info(f"Function result: {result}")
        
    except Exception as e:
        logger.error(f"Error in module function test: {str(e)}")

# Test with custom unpickler
def test_custom_unpickler():
    logger.info("Testing with custom unpickler")
    model_path = os.path.join("attached_assets", "disease_prediction_model.pkl")
    
    try:
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'numpy._core':
                    module = 'numpy.core'
                return super().find_class(module, name)
        
        with open(model_path, "rb") as f:
            model = CustomUnpickler(f).load()
        
        logger.info("Successfully loaded model with CustomUnpickler")
        
        # Test the model works
        from models.disease import ALL_SYMPTOMS, process_symptoms
        test_symptoms = ["headache", "nausea", "high_fever"]
        input_vector = process_symptoms(test_symptoms)
        
        prediction = model.predict([input_vector])
        probabilities = model.predict_proba([input_vector])
        logger.info(f"Custom unpickler prediction: {prediction[0]}")
        logger.info(f"Confidence: {max(probabilities[0]) * 100:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in custom unpickler test: {str(e)}")

if __name__ == "__main__":
    print("=== Testing Disease Prediction Model ===")
    test_direct_load()
    print("\n=== Testing Predict Disease Function ===")
    test_module_function()
    print("\n=== Testing Custom Unpickler ===")
    test_custom_unpickler()