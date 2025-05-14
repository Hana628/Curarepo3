import re
import logging
import os

# Set up logger
logger = logging.getLogger(__name__)

# Import from local_chatbot 
try:
    from local_chatbot import get_chatbot_response as local_get_response
    from local_chatbot import get_welcome_message as local_get_welcome
    from local_chatbot import get_suggestion_chips as local_get_chips
    local_chatbot_available = True
    logger.info("Local chatbot module loaded successfully.")
except ImportError as e:
    logger.error(f"Local chatbot not available: {e}")
    local_chatbot_available = False

# Safely import numpy only if needed
try:
    import numpy as np
    np_available = True
except ImportError:
    logger.warning("NumPy not available")
    np_available = False

# Legacy chatbot responses (fallback if OpenAI is not available)
GREETING_PATTERNS = [
    r'hi\b', r'hello\b', r'hey\b', r'howdy\b', r'greetings\b'
]

HEALTH_QUESTION_PATTERNS = [
    r'blood pressure', r'diabetes', r'lifestyle', r'anomaly', r'health'
]

def get_welcome_message():
    """Return a welcome message for the chatbot."""
    # Use local chatbot if available
    if local_chatbot_available:
        welcome_text = local_get_welcome()
        suggestions = local_get_chips()
        
        return {
            "text": welcome_text,
            "options": suggestions
        }
    else:
        # Fallback to legacy welcome message
        return {
            "text": """Hello! I'm your CURA healthcare assistant. How can I help you today?
            
I can assist with:
• Blood pressure prediction
• Diabetes risk assessment 
• Lifestyle recommendations
• Health anomaly detection""",
            "options": [
                "Check my blood pressure",
                "Assess my diabetes risk",
                "Get lifestyle recommendations",
                "Detect health anomalies"
            ]
        }

def get_suggestion_chips():
    """Return suggestion chips for the chat interface."""
    if local_chatbot_available:
        return local_get_chips()
    else:
        return [
            "Check my blood pressure",
            "Assess my diabetes risk",
            "Get lifestyle recommendations",
            "Detect health anomalies"
        ]

def get_chatbot_response(user_message):
    """Generate a response based on the user's message."""
    # First try to use local chatbot
    if local_chatbot_available:
        try:
            # Get response from local chatbot
            response = local_get_response(user_message)
            
            # Check if there's a suggested action
            action = None
            if response.get("suggested_action") == "showDiabetesForm":
                action = "showDiabetesForm"
            elif response.get("suggested_action") == "showBloodPressureForm":
                action = "showBloodPressureForm"
            elif response.get("suggested_action") == "showLifestyleForm":
                action = "showLifestyleForm"
            elif response.get("suggested_action") == "showAnomalyForm":
                action = "showAnomalyForm"
            elif response.get("suggested_action") == "showAppointmentForm":
                action = "showAppointmentForm"
            elif response.get("suggested_action") == "showPredictionOptions":
                action = "showPredictionOptions"
            
            # Return formatted response
            return {
                "text": response.get("text"),
                "action": action,
                "options": get_suggestion_chips() if not action else None
            }
        except Exception as e:
            logger.error(f"Error calling local chatbot: {str(e)}")
            # Fall back to legacy response if there's an error
            return get_legacy_chatbot_response(user_message)
    
    # Fall back to legacy response
    return get_legacy_chatbot_response(user_message)

def get_legacy_chatbot_response(user_message):
    """Legacy response generation (without OpenAI)."""
    user_message = user_message.lower().strip()
    
    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, user_message):
            return {
                "text": "Hello! I'm your CURA healthcare assistant. How can I help you today?",
                "options": [
                    "Check my blood pressure",
                    "Assess my diabetes risk", 
                    "Get lifestyle recommendations",
                    "Detect health anomalies"
                ]
            }
    
    # Check for health-related questions
    if "blood pressure" in user_message:
        return {
            "text": "I can help predict your blood pressure risk based on health metrics. Would you like to proceed with the blood pressure assessment?",
            "action": "showBloodPressureForm"
        }
    elif "diabetes" in user_message:
        return {
            "text": "I can assess your diabetes risk using our prediction model. Shall we proceed with the diabetes risk assessment?",
            "action": "showDiabetesForm"
        }
    elif "lifestyle" in user_message:
        return {
            "text": "I can provide personalized lifestyle recommendations based on your health profile. Would you like to proceed?",
            "action": "showLifestyleForm"
        }
    elif "anomaly" in user_message:
        return {
            "text": "I can analyze your health data to detect any anomalies or patterns that might need attention. Would you like to proceed with the analysis?",
            "action": "showAnomalyForm"
        }
    elif any(word in user_message for word in ["help", "what", "how", "can you"]):
        return get_welcome_message()
    
    # Default response
    return {
        "text": "I'm not sure I understand. As your CURA assistant, I can help with blood pressure prediction, diabetes risk assessment, lifestyle recommendations, or health anomaly detection. How can I assist you?",
        "options": [
            "Check my blood pressure",
            "Assess my diabetes risk",
            "Get lifestyle recommendations",
            "Detect health anomalies"
        ]
    }

def validate_bp_input(data):
    """Validate input data for blood pressure prediction."""
    required_fields = ['age', 'weight', 'height', 'gender', 'smoking', 'alcohol', 'exercise']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return {
                'status': 'error',
                'message': f"Missing required field: {field}"
            }
    
    # Check data types and ranges
    try:
        age = float(data['age'])
        if age < 18 or age > 100:
            return {
                'status': 'error',
                'message': "Age must be between 18 and 100"
            }
        
        weight = float(data['weight'])
        if weight < 30 or weight > 300:
            return {
                'status': 'error',
                'message': "Weight must be between 30kg and 300kg"
            }
        
        height = float(data['height'])
        if height < 100 or height > 250:
            return {
                'status': 'error',
                'message': "Height must be between 100cm and 250cm"
            }
    except ValueError:
        return {
            'status': 'error',
            'message': "Numeric values must be valid numbers"
        }
    
    return {'status': 'success'}

def validate_diabetes_input(data):
    """Validate input data for diabetes prediction."""
    required_fields = ['age', 'glucose', 'bmi', 'family_history', 'physical_activity']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return {
                'status': 'error',
                'message': f"Missing required field: {field}"
            }
    
    # Check data types and ranges
    try:
        age = float(data['age'])
        if age < 18 or age > 100:
            return {
                'status': 'error',
                'message': "Age must be between 18 and 100"
            }
        
        glucose = float(data['glucose'])
        if glucose < 60 or glucose > 300:
            return {
                'status': 'error',
                'message': "Glucose level must be between 60 and 300 mg/dL"
            }
        
        bmi = float(data['bmi'])
        if bmi < 15 or bmi > 60:
            return {
                'status': 'error',
                'message': "BMI must be between 15 and 60"
            }
    except ValueError:
        return {
            'status': 'error',
            'message': "Numeric values must be valid numbers"
        }
    
    return {'status': 'success'}

def validate_lifestyle_input(data):
    """Validate input data for lifestyle recommendations."""
    required_fields = ['age', 'weight', 'height', 'activity_level', 'diet_preference', 'sleep_hours']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return {
                'status': 'error',
                'message': f"Missing required field: {field}"
            }
    
    # Check data types and ranges
    try:
        age = float(data['age'])
        if age < 18 or age > 100:
            return {
                'status': 'error',
                'message': "Age must be between 18 and 100"
            }
        
        weight = float(data['weight'])
        if weight < 30 or weight > 300:
            return {
                'status': 'error',
                'message': "Weight must be between 30kg and 300kg"
            }
        
        height = float(data['height'])
        if height < 100 or height > 250:
            return {
                'status': 'error',
                'message': "Height must be between 100cm and 250cm"
            }
        
        sleep_hours = float(data['sleep_hours'])
        if sleep_hours < 0 or sleep_hours > 24:
            return {
                'status': 'error',
                'message': "Sleep hours must be between 0 and 24"
            }
    except ValueError:
        return {
            'status': 'error',
            'message': "Numeric values must be valid numbers"
        }
    
    activity_levels = ['sedentary', 'light', 'moderate', 'active', 'very_active']
    if data['activity_level'] not in activity_levels:
        return {
            'status': 'error',
            'message': f"Activity level must be one of: {', '.join(activity_levels)}"
        }
    
    diet_preferences = ['omnivore', 'vegetarian', 'vegan', 'pescatarian', 'keto', 'paleo']
    if data['diet_preference'] not in diet_preferences:
        return {
            'status': 'error',
            'message': f"Diet preference must be one of: {', '.join(diet_preferences)}"
        }
    
    return {'status': 'success'}

def validate_anomaly_input(data):
    """Validate input data for anomaly detection."""
    if 'health_metrics' not in data or not isinstance(data['health_metrics'], list):
        return {
            'status': 'error',
            'message': "Missing or invalid health metrics data. Please provide a list of health metric values."
        }
    
    health_metrics = data['health_metrics']
    
    # Check if we have enough data points
    if len(health_metrics) < 7:
        return {
            'status': 'error',
            'message': "Please provide at least 7 data points for anomaly detection."
        }
    
    # Check if all data points are valid numbers
    try:
        for metric in health_metrics:
            float(metric)
    except (ValueError, TypeError):
        return {
            'status': 'error',
            'message': "All health metrics must be valid numbers."
        }
    
    return {'status': 'success'}

def validate_disease_input(data):
    """Validate input data for disease prediction."""
    required_fields = ['symptoms']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return {
                'status': 'error',
                'message': f"Missing required field: {field}"
            }
    
    # Check that symptoms is a non-empty array
    symptoms = data.get('symptoms', [])
    if not isinstance(symptoms, list) or len(symptoms) < 1:
        return {
            'status': 'error',
            'message': "At least one symptom must be selected"
        }
    
    # Check that all symptoms are strings
    if not all(isinstance(s, str) for s in symptoms):
        return {
            'status': 'error',
            'message': "All symptoms must be text values"
        }
    
    return {'status': 'success'}

def validate_skin_disease_input(data):
    """Validate input data for skin disease prediction."""
    # Check if either image or image_data is present
    if not data.get('image') and not data.get('image_data'):
        return {
            'status': 'error',
            'message': "An image must be provided"
        }
    
    # If image_data is provided, check if it's a string (base64)
    image_data = data.get('image_data')
    if image_data and not isinstance(image_data, str):
        return {
            'status': 'error',
            'message': "Image data must be a base64 encoded string"
        }
    
    # If image is provided, check if it's a valid file object (for form uploads)
    # Note: This validation happens in the route handler
    
    # Optional fields validation
    location = data.get('location')
    if location and not isinstance(location, str):
        return {
            'status': 'error',
            'message': "Location must be a text value"
        }
    
    duration = data.get('duration')
    if duration and not isinstance(duration, str):
        return {
            'status': 'error',
            'message': "Duration must be a text value"
        }
    
    # Check that symptoms is an array of strings if provided
    symptoms = data.get('symptoms', [])
    if symptoms and (not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms)):
        return {
            'status': 'error',
            'message': "Symptoms must be an array of text values"
        }
    
    return {'status': 'success'}
