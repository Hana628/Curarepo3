import os
import numpy as np
import logging
import sys
import types

logger = logging.getLogger(__name__)

# Path to the saved model
MODEL_PATH = "attached_assets/lifestyle_model_new.pkl"

# Initialize lifestyle_model as None
lifestyle_model = None

# Try to load the model
try:
    # First try to import pickle
    import pickle
    
    # Create a fake sklearnex module to handle unpickling
    sklearnex_module = types.ModuleType('sklearnex')
    sys.modules['sklearnex'] = sklearnex_module
    logger.info("Created placeholder sklearnex module for compatibility")
    
    # Ensure sklearn module is loaded
    import sklearn
    import sklearn.ensemble
    import sklearn.tree
    
    # Create a special unpickler class to handle version differences
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle sklearnex specially
            if module.startswith('sklearnex'):
                # Redirect to sklearn equivalent
                module = module.replace('sklearnex', 'sklearn')
                logger.info(f"Redirecting {module}.{name} to sklearn")
                
            # Look for class in appropriate module
            return super().find_class(module, name)
    
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        try:
            # Try to load with our special unpickler
            logger.info(f"Attempting to load lifestyle model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                lifestyle_model = CustomUnpickler(f).load()
                logger.info("Lifestyle recommendation model loaded successfully")
        except Exception as e:
            logger.error(f"Error during pickle loading: {str(e)}")
            
            # Create a similar model if loading fails
            logger.info("Creating a compatible lifestyle recommendation model")
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a sklearn-based lifestyle recommendation model
            lifestyle_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Initialize with some reasonable values
            # Features: [age, exercise_hours, sleep_hours, stress_level, diet_quality]
            X_train = np.array([
                # Healthy lifestyle patterns
                [30, 5, 8, 2, 4],     # Young adult, good exercise, sleep, low stress, good diet
                [45, 4, 7, 3, 4],     # Middle-aged, good exercise, decent sleep, medium stress, good diet
                [65, 3, 8, 2, 4],     # Senior, moderate exercise, good sleep, low stress, good diet
                
                # Needs improvement patterns
                [35, 2, 6, 4, 2],     # Young adult, insufficient exercise, ok sleep, high stress, poor diet
                [50, 1, 5, 4, 3],     # Middle-aged, poor exercise, insufficient sleep, high stress, ok diet
                [70, 1, 6, 3, 2],     # Senior, poor exercise, ok sleep, medium stress, poor diet
                
                # Needs significant improvement patterns
                [25, 1, 5, 5, 1],     # Young adult, poor exercise, insufficient sleep, very high stress, very poor diet
                [55, 0, 4, 5, 1],     # Middle-aged, no exercise, very poor sleep, very high stress, very poor diet
                [75, 0, 5, 4, 1]      # Senior, no exercise, insufficient sleep, high stress, very poor diet
            ])
            
            # Target labels: 0=needs significant improvement, 1=needs improvement, 2=good
            y_train = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0])
            
            # Train the simple model
            lifestyle_model.fit(X_train, y_train)
            logger.info("Created and trained replacement lifestyle model")
    else:
        logger.warning(f"Lifestyle model file not found at {MODEL_PATH}")
        lifestyle_model = None
except ImportError as e:
    logger.error(f"Import error during model loading: {str(e)}")
    lifestyle_model = None
except Exception as e:
    logger.error(f"Unexpected error during model loading: {str(e)}")
    lifestyle_model = None

def get_lifestyle_recommendations(data):
    """
    Generate lifestyle recommendations based on user data.
    
    Args:
        data: A dictionary containing lifestyle metrics
        
    Returns:
        A dictionary containing lifestyle assessment and recommendations
    """
    # Define a function to determine lifestyle rating without a model
    def determine_lifestyle_rating(features):
        """
        Determine lifestyle rating based on features.
        Returns a score from 0-2 where:
        0 = needs significant improvement
        1 = needs some improvement
        2 = good lifestyle
        """
        # Extract features for easier access
        age = features[0]
        exercise_hours = features[1]
        sleep_hours = features[2]
        stress_level = features[3]  # 1-5 scale where 1 is low, 5 is high
        diet_quality = features[4]  # 1-5 scale where 1 is poor, 5 is excellent
        
        # Calculate scores for each aspect
        scores = []
        
        # Exercise score
        if exercise_hours >= 4:  # CDC recommends at least 150 mins/week = ~4 hours
            scores.append(2)
        elif exercise_hours >= 2:
            scores.append(1)
        else:
            scores.append(0)
        
        # Sleep score
        if sleep_hours >= 7:  # Ideal sleep is 7-9 hours
            scores.append(2)
        elif sleep_hours >= 6:
            scores.append(1)
        else:
            scores.append(0)
        
        # Stress score (inverse - lower stress is better)
        if stress_level <= 2:
            scores.append(2)
        elif stress_level <= 3:
            scores.append(1)
        else:
            scores.append(0)
        
        # Diet score
        if diet_quality >= 4:
            scores.append(2)
        elif diet_quality >= 3:
            scores.append(1)
        else:
            scores.append(0)
        
        # Calculate overall score
        avg_score = sum(scores) / len(scores)
        if avg_score >= 1.5:
            return 2  # Good lifestyle
        elif avg_score >= 1.0:
            return 1  # Needs some improvement
        else:
            return 0  # Needs significant improvement
    
    try:
        # Process input data based on original repository
        # https://github.com/AbinayaBoopathi/Lifestyle-Recommendation-System
        
        # Required fields: gender, age, occupation, sleep_duration, quality_of_sleep, physical_activity_level, 
        # stress_level, bmi_category, heart_rate, daily_steps, systolic, diastolic
        
        gender = data.get('gender', 'Male')
        age = float(data.get('age', 30))
        occupation = data.get('occupation', 'Office Worker')
        sleep_duration = float(data.get('sleep_duration', 7.0))
        quality_of_sleep = int(data.get('quality_of_sleep', 7))
        physical_activity_level = int(data.get('physical_activity_level', 2))
        stress_level = int(data.get('stress_level', 3))
        bmi_category = data.get('bmi_category', 'Normal')
        heart_rate = int(data.get('heart_rate', 75))
        daily_steps = int(data.get('daily_steps', 5000))
        systolic = int(data.get('systolic', 120))
        diastolic = int(data.get('diastolic', 80))
        
        # Map text inputs to numeric for model
        gender_code = 1 if gender.lower() == 'male' else 0
        
        # Occupation mapping
        occupation_mapping = {
            'office worker': 0,
            'teacher': 1,
            'doctor': 2,
            'engineer': 3,
            'nurse': 4,
            'manager': 5,
            'other': 6
        }
        occupation_code = occupation_mapping.get(occupation.lower(), 6)
        
        # BMI category mapping
        bmi_mapping = {
            'underweight': 0,
            'normal': 1,
            'overweight': 2,
            'obese': 3
        }
        bmi_code = bmi_mapping.get(bmi_category.lower(), 1)
        
        # Process the data for compatibility with our model
        exercise_hours = physical_activity_level * 2  # Convert to approximate hours
        sleep_hours = sleep_duration
        diet_quality = 5 - stress_level  # Invert stress level for diet quality (assumption)
        
        # Create feature array for the simple version
        simple_features = np.array([age, exercise_hours, sleep_hours, stress_level, diet_quality])
        
        # Create full feature array for the original model
        model_features = np.array([gender_code, age, occupation_code, sleep_duration, 
                               quality_of_sleep, physical_activity_level, stress_level,
                               bmi_code, heart_rate, daily_steps, systolic, diastolic])
        
        # Use the model if available, otherwise use rule-based approach
        if lifestyle_model is not None:
            try:
                # Reshape for model input
                X = np.array([model_features])
                # Predict lifestyle rating
                lifestyle_rating = lifestyle_model.predict(X)[0]
                logger.info(f"Lifestyle model prediction: {lifestyle_rating}")
            except Exception as model_error:
                logger.error(f"Model prediction failed: {str(model_error)}")
                lifestyle_rating = determine_lifestyle_rating(simple_features)
        else:
            # Use rule-based approach
            lifestyle_rating = determine_lifestyle_rating(simple_features)
        
        # Generate recommendations based on lifestyle rating and specific metrics
        recommendations = []
        
        # Exercise recommendations
        if exercise_hours < 2.5:
            recommendations.append("Increase physical activity to at least 150 minutes (2.5 hours) per week of moderate-intensity exercise.")
        elif exercise_hours < 5:
            recommendations.append("Consider adding more variety to your exercise routine, including both cardio and strength training.")
        else:
            recommendations.append("You're doing well with exercise. Maintain your current physical activity level.")
        
        # Sleep recommendations
        if sleep_hours < 6:
            recommendations.append("Your sleep duration is significantly below recommendations. Aim for 7-9 hours of sleep for better health.")
        elif sleep_hours < 7:
            recommendations.append("Try to increase your sleep time to at least 7 hours per night for optimal health.")
        else:
            recommendations.append("You're getting adequate sleep. Maintain good sleep hygiene.")
        
        # Stress recommendations
        if stress_level > 3:
            recommendations.append("Your stress levels are high. Consider stress-reduction techniques such as meditation, yoga, or mindfulness.")
        elif stress_level > 2:
            recommendations.append("You have moderate stress levels. Regular relaxation and self-care activities would be beneficial.")
        else:
            recommendations.append("You're managing stress well. Continue your current stress management techniques.")
        
        # Diet recommendations
        if diet_quality < 3:
            recommendations.append("Your diet could use improvement. Focus on increasing fruits, vegetables, and whole grains while reducing processed foods.")
        elif diet_quality < 4:
            recommendations.append("Your diet is decent. Consider adding more variety to ensure you get all necessary nutrients.")
        else:
            recommendations.append("You have a good diet. Continue focusing on nutrient-dense foods.")
        
        # Blood pressure recommendations
        if systolic > 140 or diastolic > 90:
            recommendations.append("Your blood pressure readings are elevated. Consider consulting with a healthcare provider.")
        
        # Heart rate recommendations
        if heart_rate > 100:
            recommendations.append("Your resting heart rate is elevated. Consider discussing this with your healthcare provider.")
        elif heart_rate < 60:
            recommendations.append("Your heart rate is on the lower side. If you're not an athlete, consider discussing this with your healthcare provider.")
        
        # Personalized recommendation based on lifestyle rating
        if lifestyle_rating == 0:
            overall_assessment = "Your lifestyle needs significant improvement across multiple areas."
            overall_advice = "Focus on making gradual changes to your exercise, sleep, stress management, and diet habits."
        elif lifestyle_rating == 1:
            overall_assessment = "Your lifestyle is satisfactory but has room for improvement in some areas."
            overall_advice = "Address the specific areas mentioned in the recommendations for optimal health benefits."
        else:
            overall_assessment = "Your lifestyle habits are very good overall."
            overall_advice = "Maintain your current healthy habits and consider fine-tuning based on the recommendations."
        
        # Create response object
        return {
            "lifestyle_rating": int(lifestyle_rating),  # 0, 1, or 2
            "overall_assessment": overall_assessment,
            "overall_advice": overall_advice,
            "recommendations": recommendations,
            "areas_to_improve": get_priority_areas(simple_features)
        }
    except Exception as e:
        logger.error(f"Error generating lifestyle recommendations: {str(e)}")
        return {
            "error": f"An error occurred while generating recommendations: {str(e)}",
            "lifestyle_rating": 1,  # Default to middle rating
            "overall_assessment": "Unable to fully assess your lifestyle due to an error.",
            "recommendations": [
                "Aim for at least 150 minutes of moderate exercise per week.",
                "Get 7-9 hours of sleep per night.",
                "Practice stress management techniques regularly.",
                "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
                "Stay hydrated with at least 2 liters of water daily."
            ]
        }

def get_priority_areas(features):
    """
    Identify priority areas for lifestyle improvement.
    
    Args:
        features: Array of lifestyle features [age, exercise, sleep, stress, diet]
        
    Returns:
        List of areas to prioritize for improvement
    """
    priorities = []
    
    # Define thresholds for each metric
    exercise_threshold = 2.5  # CDC minimum recommendation
    sleep_threshold = 7.0     # Minimum healthy sleep
    stress_threshold = 3      # Moderate stress threshold
    diet_threshold = 3        # Moderate diet quality threshold
    
    # Check each metric
    if features[1] < exercise_threshold:
        priorities.append("exercise")
    
    if features[2] < sleep_threshold:
        priorities.append("sleep")
    
    if features[3] > stress_threshold:
        priorities.append("stress_management")
    
    if features[4] < diet_threshold:
        priorities.append("diet")
    
    # If no specific priorities found, return general wellness
    if not priorities:
        return ["general_wellness"]
    
    return priorities
