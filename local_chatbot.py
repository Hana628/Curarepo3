import logging
from medical_chatbot import chatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chatbot_response(user_message):
    """
    Generate a response for the healthcare chatbot using our medical chatbot model.
    This implementation mimics the approach from the GitHub repo but without dependency issues.
    
    Args:
        user_message (str): The user's message.
    
    Returns:
        dict: A dictionary containing the response text and any suggested actions.
    """
    try:
        # Get response from medical chatbot
        result = chatbot.get_response(user_message)
        
        # Get the response text
        response_text = result.get("response", "I'm sorry, I couldn't understand that.")
        
        # Check the context for suggested actions
        context = result.get("context", "")
        suggested_action = None
        
        # Map contexts to actions
        context_to_action = {
            "blood_pressure_prediction": "showBloodPressureForm",
            "diabetes_prediction": "showDiabetesForm",
            "lifestyle_recommendation": "showLifestyleForm",
            "anomaly_detection": "showAnomalyForm",
            "disease_prediction": "showDiseaseForm",
            "skin_disease_prediction": "showSkinDiseaseForm",
            "emergency": "showEmergencyInfo"
        }
        
        # Set the suggested action based on context
        if context in context_to_action:
            suggested_action = context_to_action[context]
        
        return {
            "text": response_text,
            "suggested_action": suggested_action
        }
    
    except Exception as e:
        logger.error(f"Error generating chatbot response: {str(e)}")
        return {
            "text": "I'm having trouble processing your request. Please try again or ask a different question."
        }

def get_welcome_message():
    """
    Generate a welcome message for new users.
    
    Returns:
        str: A welcome message introducing the chatbot.
    """
    try:
        return chatbot.get_welcome_message()
    except Exception as e:
        logger.error(f"Error generating welcome message: {str(e)}")
        return "Hello! I'm your CURA health assistant. How can I help you today?"

def get_suggestion_chips():
    """
    Generate suggestion chips for the chat interface.
    
    Returns:
        list: A list of suggested queries for the user.
    """
    try:
        return chatbot.get_suggestion_chips()
    except Exception as e:
        logger.error(f"Error generating suggestion chips: {str(e)}")
        return [
            "What can you do?",
            "Blood pressure check",
            "Diabetes risk",
            "Lifestyle advice",
            "I have a headache"
        ]
