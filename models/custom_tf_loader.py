import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def load_tf_model_safely(model_path):
    """
    Loads a TensorFlow model from the given path and handles errors.
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at: {model_path}")
        return None

    try:
        # Load the model with TensorFlow's standard load_model function
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Model loaded successfully using tf.keras.models.load_model")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None
