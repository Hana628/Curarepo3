import os
import logging
import base64
import sys

# Configure logging early
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Force compatible NumPy version for TensorFlow
try:
    # Try to set environment variable to downgrade NumPy API version
    os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
    # This can help with compatibility between newer NumPy and older TensorFlow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
    
    logger.info("Set NumPy compatibility environment variables")
except Exception as e:
    logger.warning(f"Failed to set environment variables: {str(e)}")

try:
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
    from werkzeug.middleware.proxy_fix import ProxyFix
    from werkzeug.utils import secure_filename
except ImportError as e:
    logger.error(f"Failed to import Flask dependencies: {str(e)}")
    sys.exit(1)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env file loading")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# Try to import model functions, but handle errors gracefully
try:
    from models.blood_pressure import predict_blood_pressure
    bp_available = True
except Exception as e:
    logger.error(f"Blood pressure model import failed: {str(e)}")
    bp_available = False
    
    def predict_blood_pressure(data):
        return {
            "error": "Blood pressure module unavailable. Please check logs for details.",
            "risk_category": "Unknown",
            "advice": "We're experiencing technical difficulties. Please try again later.",
            "systolic_estimate": 120,
            "diastolic_estimate": 80
        }

try:
    from models.diabetes import predict_diabetes
    diabetes_available = True
except Exception as e:
    logger.error(f"Diabetes model import failed: {str(e)}")
    diabetes_available = False
    
    def predict_diabetes(data):
        return {
            "error": "Diabetes module unavailable. Please check logs for details.",
            "risk_category": "Unknown",
            "advice": "We're experiencing technical difficulties. Please try again later.",
            "risk_score": 0.5,
            "recommendations": ["Service temporarily unavailable"]
        }

try:
    from models.lifestyle import get_lifestyle_recommendations
    lifestyle_available = True
except Exception as e:
    logger.error(f"Lifestyle model import failed: {str(e)}")
    lifestyle_available = False
    
    def get_lifestyle_recommendations(data):
        return {
            "error": "Lifestyle module unavailable. Please check logs for details.",
            "bmi": 25.0,
            "weight_status": "unknown",
            "sleep_quality": "unknown",
            "sleep_hours": float(data.get('sleep_hours', 7)),
            "exercise_recommendations": ["Service temporarily unavailable"],
            "diet_recommendations": ["Service temporarily unavailable"],
            "sleep_recommendations": ["Service temporarily unavailable"],
            "weight_recommendations": ["Service temporarily unavailable"],
            "general_recommendations": ["Service temporarily unavailable"]
        }

try:
    from models.anomaly import detect_anomalies
    anomaly_available = True
except Exception as e:
    logger.error(f"Anomaly model import failed: {str(e)}")
    anomaly_available = False
    
    def detect_anomalies(data):
        return {
            "error": "Anomaly detection module unavailable. Please check logs for details.",
            "message": "Service temporarily unavailable",
            "anomalies_detected": False,
            "anomaly_indices": [],
            "anomaly_scores": [],
            "data_points": len(data.get('health_metrics', []))
        }

# Import the disease prediction model
try:
    from models.disease import predict_disease
    disease_available = True
except Exception as e:
    logger.error(f"Disease prediction model import failed: {str(e)}")
    disease_available = False
    
    def predict_disease(symptoms):
        return {
            "error": "Disease prediction module unavailable. Please check logs for details.",
            "disease": "Unknown",
            "confidence": 0,
            "specialist": "general physician",
            "description": "We're experiencing technical difficulties. Please try again later.",
            "care_recommendations": ["Consult a healthcare professional for proper diagnosis."]
        }

# Import the skin disease prediction model
try:
    from models.skin_disease import predict_skin_disease
    skin_disease_available = True
except Exception as e:
    logger.error(f"Skin disease prediction model import failed: {str(e)}")
    skin_disease_available = False
    
    def predict_skin_disease(image_data, location=None, duration=None, symptoms=None):
        return {
            "error": "Skin disease prediction module unavailable. Please check logs for details.",
            "disease": "Unknown skin condition",
            "confidence": 0,
            "specialist": "dermatologist",
            "description": "We're experiencing technical difficulties. Please try again later.",
            "care_recommendations": ["Consult a dermatologist for proper diagnosis."]
        }

# Load chatbot responses
from utils import (
    get_chatbot_response, 
    get_welcome_message, 
    validate_bp_input, 
    validate_diabetes_input,
    validate_lifestyle_input,
    validate_anomaly_input,
    validate_disease_input,
    validate_skin_disease_input
)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/simple-chat')
def simple_chat():
    """Render the standalone chat page."""
    return render_template('chat.html')

@app.route('/chat-only')
def chat_only():
    """Render just the chatbot interface."""
    return render_template('chat_only.html')

@app.route('/chat-standalone')
def chat_standalone():
    """Render the redesigned standalone chatbot interface."""
    return render_template('chat_standalone.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process user chat messages and return a response."""
    try:
        data = request.json or {}
        user_message = data.get('message', '')
        
        # Get response from chatbot logic
        response = get_chatbot_response(user_message)
        
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

@app.route('/get_welcome_message', methods=['GET'])
def welcome_message():
    """Return the welcome message for the chatbot."""
    try:
        message = get_welcome_message()
        return jsonify({
            'status': 'success',
            'message': message
        })
    except Exception as e:
        logger.error(f"Error fetching welcome message: {str(e)}")
        # Provide a fallback welcome message if there's an error
        fallback_message = {
            "text": """## 👋 Welcome to CURA Health Assistant!
            
            I'm here to help with your health questions and concerns. Here are some ways I can assist you:
            
            * Answer general health questions
            * Help you navigate the CURA app
            * Provide health and wellness information
            * Direct you to the right features for your needs
            
            Feel free to ask me anything related to your health!""",
            "options": [
                "Check my blood pressure",
                "Assess my diabetes risk",
                "Get lifestyle recommendations",
                "Detect health anomalies"
            ]
        }
        return jsonify({
            'status': 'success',
            'message': fallback_message
        })

@app.route('/predict/blood_pressure', methods=['POST'])
def blood_pressure_prediction():
    """Process blood pressure prediction requests."""
    try:
        data = request.json
        
        # Validate input data
        validation_result = validate_bp_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        # Make prediction
        prediction = predict_blood_pressure(data)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        logger.error(f"Error in blood pressure prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

@app.route('/predict/diabetes', methods=['POST'])
def diabetes_prediction():
    """Process diabetes prediction requests."""
    try:
        data = request.json
        
        # Validate input data
        validation_result = validate_diabetes_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        # Make prediction
        prediction = predict_diabetes(data)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

@app.route('/recommend/lifestyle', methods=['POST'])
def lifestyle_recommendation():
    """Process lifestyle recommendation requests."""
    try:
        data = request.json
        
        # Validate input data
        validation_result = validate_lifestyle_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        # Get recommendations
        recommendations = get_lifestyle_recommendations(data)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in lifestyle recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

@app.route('/detect/anomaly', methods=['POST'])
def anomaly_detection():
    """Process anomaly detection requests."""
    try:
        data = request.json
        
        # Validate input data
        validation_result = validate_anomaly_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        # Detect anomalies
        result = detect_anomalies(data)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

# Disease prediction endpoint
@app.route('/predict/disease', methods=['POST'])
def disease_prediction_endpoint():
    """Process disease prediction requests based on symptoms."""
    try:
        # Get symptom data
        data = request.json
        
        # Validate input data
        validation_result = validate_disease_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        symptoms = data.get('symptoms', []) if data else []
        
        # Use the disease prediction model
        if disease_available:
            prediction = predict_disease(symptoms)
            
            return jsonify({
                'status': 'success',
                'prediction': prediction
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Disease prediction is currently unavailable. Please try again later.'
            }), 503
            
    except Exception as e:
        logger.error(f"Error in disease prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

# Skin disease detection endpoint
@app.route('/predict/skin_disease', methods=['POST'])
def skin_disease_prediction():
    """Process skin disease prediction requests using the integrated model."""
    try:
        # Prepare data for validation
        data = {}
        
        # Process image from file or JSON
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_data = base64.b64encode(file.read()).decode('utf-8')
                data['image_data'] = image_data
        elif request.json and 'image_data' in request.json:
            data['image_data'] = request.json.get('image_data', '')
            
        # Add additional parameters
        data['location'] = request.form.get('location') if request.form else (request.json.get('location') if request.json else None)
        data['duration'] = request.form.get('duration') if request.form else (request.json.get('duration') if request.json else None)
        data['symptoms'] = request.form.getlist('symptoms') if request.form else (request.json.get('symptoms', []) if request.json else [])
        
        # Validate input data
        validation_result = validate_skin_disease_input(data)
        if validation_result['status'] == 'error':
            return jsonify(validation_result), 400
        
        # Get the image data for processing
        image_data = data['image_data']
        
        # Get additional information
        location = request.form.get('location') if request.form else (request.json.get('location') if request.json else None)
        duration = request.form.get('duration') if request.form else (request.json.get('duration') if request.json else None)
        symptoms = request.form.getlist('symptoms') if request.form else (request.json.get('symptoms', []) if request.json else [])
        
        # Use our skin disease prediction model
        if skin_disease_available:
            prediction = predict_skin_disease(image_data, location, duration, symptoms)
            
            return jsonify({
                'status': 'success',
                'prediction': prediction
            })
        
        # We no longer use OpenAI as a fallback - only use the local model
        logger.warning("Skin disease model is unavailable and OpenAI fallback has been removed")
        
        # Final fallback response if both methods fail
        return jsonify({
            'status': 'error',
            'message': 'Skin disease analysis is currently unavailable. Please try again later.'
        }), 503
    except Exception as e:
        logger.error(f"Error in skin disease prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"An error occurred: {str(e)}"
        }), 500

@app.route('/health-monitor')
def health_monitor():
    """Render the health monitoring dashboard."""
    # Pass a special flag to prevent chat from loading
    return render_template('health_monitor.html', hide_chat=True)

@app.route('/api/health-data')
def get_health_data():
    """API endpoint to get simulated health data from wearable devices."""
    # In a real application, this would connect to actual wearable devices or a database
    # For now, we'll simulate the data
    
    import random
    import math
    from datetime import datetime, timedelta
    
    # Generate time series for the last 24 hours
    now = datetime.now()
    times = []
    heart_rate_values = []
    
    # Base pattern with diurnal variation (lower at night, higher during day)
    for i in range(24):
        time_point = now - timedelta(hours=23-i)
        times.append(time_point.strftime('%H:%M'))
        
        # Heart rate varies between 60-100 with diurnal pattern
        hour = time_point.hour
        # Lower at night (11pm-6am), higher during day with peak at 5-6pm
        if 23 <= hour or hour < 6:
            base_hr = 65  # Sleeping/resting
        elif 6 <= hour < 9:
            base_hr = 70 + (hour - 6) * 3  # Waking up, gradually increasing
        elif 9 <= hour < 17:
            base_hr = 80  # Daytime
        elif 17 <= hour < 19:
            base_hr = 85  # Peak activity
        else:
            base_hr = 80 - (hour - 19) * 2  # Evening, gradually decreasing
            
        # Add some variation
        variation = math.sin(i/4) * 8 + random.uniform(-5, 5)
        heart_rate = round(base_hr + variation)
        heart_rate_values.append(heart_rate)
    
    # Get min, max, current values
    min_hr = min(heart_rate_values)
    max_hr = max(heart_rate_values)
    current_hr = heart_rate_values[-1]
    
    # Determine heart rate status
    if current_hr < 60:
        hr_status = "Low"
    elif current_hr > 100:
        hr_status = "Elevated"
    else:
        hr_status = "Normal"
    
    # Generate blood pressure data
    # Systolic: 90-140, Diastolic: 60-90
    systolic = random.randint(110, 135)
    diastolic = random.randint(70, 85)
    
    # Determine BP status
    if systolic < 120 and diastolic < 80:
        bp_status = "Normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        bp_status = "Elevated"
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        bp_status = "High (Stage 1)"
    else:
        bp_status = "High (Stage 2)"
    
    # Generate oxygen data (95-100%)
    oxygen_value = round(random.uniform(95, 99), 1)
    oxygen_status = "Normal" if oxygen_value >= 95 else "Low"
    
    # Generate glucose data (70-140 mg/dL is normal)
    glucose_value = round(random.uniform(80, 130))
    if glucose_value < 70:
        glucose_status = "Low"
    elif glucose_value > 140:
        glucose_status = "Elevated"
    else:
        glucose_status = "Normal"
    
    # Generate activity data
    steps = random.randint(3000, 12000)
    active_minutes = random.randint(20, 90)
    goal_percentage = min(100, round((steps / 10000) * 100))
    
    # Construct the response
    response = {
        'times': times,
        'heart_rate': {
            'values': heart_rate_values,
            'current': current_hr,
            'min': min_hr,
            'max': max_hr,
            'status': hr_status
        },
        'blood_pressure': {
            'systolic': systolic,
            'diastolic': diastolic,
            'status': bp_status
        },
        'oxygen': {
            'value': oxygen_value,
            'status': oxygen_status
        },
        'glucose': {
            'value': glucose_value,
            'status': glucose_status
        },
        'activity': {
            'steps': steps,
            'active_minutes': active_minutes,
            'goal_percentage': goal_percentage
        }
    }
    
    return jsonify(response)

@app.route('/api/health-alerts')
def get_health_alerts():
    """API endpoint to get proactive health alerts based on data analysis."""
    # In a real application, this would analyze trends in the health data
    # and generate personalized alerts. For now, we'll simulate them.
    
    import random
    from datetime import datetime, timedelta
    
    # Random seed based on day to keep alerts consistent throughout the day
    day_of_year = datetime.now().timetuple().tm_yday
    random.seed(day_of_year)
    
    alert_types = ['info', 'warning', 'danger', 'success']
    alerts = []
    
    # Probability of having alerts (70%)
    if random.random() < 0.7:
        # Generate 1-3 alerts
        num_alerts = random.randint(1, 3)
        
        for i in range(num_alerts):
            # Weighted random alert type (more likely to be info or success than warning or danger)
            weights = [0.4, 0.2, 0.1, 0.3]  # info, warning, danger, success
            alert_type = random.choices(alert_types, weights=weights)[0]
            
            # Different messages based on alert type
            if alert_type == 'info':
                messages = [
                    "Your average resting heart rate is 5 bpm higher than last week.",
                    "Your sleep duration has decreased by 45 minutes compared to your monthly average.",
                    "Your step count is trending upward this week. Keep it up!",
                    "Your blood pressure readings have been more consistent this week."
                ]
                icon = "fa-info-circle"
            elif alert_type == 'warning':
                messages = [
                    "Your blood glucose level has been trending higher than normal this week.",
                    "You've been less active than usual for the past 3 days.",
                    "Your sleep pattern has been irregular for the past week.",
                    "Your heart rate variability has decreased this week, which may indicate increased stress."
                ]
                icon = "fa-exclamation-circle"
            elif alert_type == 'danger':
                messages = [
                    "Elevated blood pressure detected consistently over the past 5 readings.",
                    "Unusually low oxygen level readings detected during your sleep last night.",
                    "Your heart rate peaked at an unusually high level during yesterday's activity.",
                    "Significantly irregular heart rhythm detected in yesterday's readings."
                ]
                icon = "fa-exclamation-triangle"
            else:  # success
                messages = [
                    "You achieved your step goal for 5 consecutive days!",
                    "Your average resting heart rate has improved by 3 bpm this month.",
                    "Your sleep consistency score has improved by 15% this week.",
                    "You've reduced your sedentary time by 1 hour per day this week!"
                ]
                icon = "fa-check-circle"
            
            # Randomly select a message
            message = random.choice(messages)
            
            # Generate a time (from 15 minutes to 12 hours ago)
            minutes_ago = random.randint(15, 12 * 60)
            time_ago = datetime.now() - timedelta(minutes=minutes_ago)
            
            # Format the time
            if minutes_ago < 60:
                time_str = f"{minutes_ago} minutes ago"
            elif minutes_ago < 24 * 60:
                hours_ago = minutes_ago // 60
                time_str = f"{hours_ago} hour{'s' if hours_ago > 1 else ''} ago"
            else:
                days_ago = minutes_ago // (24 * 60)
                time_str = f"{days_ago} day{'s' if days_ago > 1 else ''} ago"
            
            alerts.append({
                'type': alert_type,
                'message': message,
                'icon': icon,
                'time': time_str
            })
    
    # Reset random seed
    random.seed(None)
    
    # Construct the response
    response = {
        'count': len(alerts),
        'alerts': alerts
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
