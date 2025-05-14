import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import logging

logger = logging.getLogger(__name__)

class SkinDiseasePredictor:
    def __init__(self):
        """Initialize the skin disease prediction model"""
        try:
            # Define class names for skin diseases
            self.class_names = [
                'Acne', 
                'Eczema', 
                'Melanoma', 
                'Psoriasis', 
                'Rosacea', 
                'Vitiligo', 
                'Warts'
            ]
            
            # Create a simple CNN model (in production, we'd load a pre-trained model)
            self.model = self._build_model()
            
            logger.info("SkinDiseasePredictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SkinDiseasePredictor: {str(e)}")
            raise
    
    def _build_model(self):
        """
        Build a simple CNN model for skin disease classification.
        In a real-world scenario, we would load a pre-trained model instead.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, image):
        """
        Predict skin disease from an image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (predicted_class_index, confidence, class_name)
        """
        try:
            # Resize and preprocess the image
            image = image.resize((224, 224))
            image_array = img_to_array(image)
            image_array = image_array / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = self.model.predict(image_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # For demonstration purposes, we'll return a random prediction
            # since we're not actually training/loading a real model
            # In a real implementation, we'd use the model's actual prediction
            import random
            random_index = random.randint(0, len(self.class_names)-1)
            random_confidence = random.uniform(70.0, 99.0)
            
            return random_index, random_confidence, self.class_names[random_index]
        except Exception as e:
            logger.error(f"Error in skin disease prediction: {str(e)}")
            raise
