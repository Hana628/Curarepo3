import os
import logging
import sys
import importlib.util

logger = logging.getLogger(__name__)

# Set NumPy compatibility environment variables
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Check for TensorFlow
tf = None
try:
    import tensorflow as tf
    logger.info("✅ TensorFlow imported successfully in loader")
except Exception as e:
    logger.error(f"❌ Failed to import TensorFlow: {e}")
    
def load_tf_model_safely(model_path):
    """
    Loads a TensorFlow model from the given path and handles errors.
    Multiple approaches are tried to ensure compatibility across versions and environments.
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at: {model_path}")
        return None
        
    if tf is None:
        logger.error("❌ TensorFlow not available, cannot load model")
        return None

    # Try multiple loading methods in order of preference
    model = None
    errors = []
    
    # Method 1: Standard Keras load_model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Model loaded successfully using tf.keras.models.load_model from {model_path}")
        return model
    except Exception as e:
        errors.append(f"Standard loading failed: {str(e)}")
        
    # Method 2: Load with custom objects
    try:
        # Define any custom objects that might be needed
        custom_objects = {}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("✅ Model loaded successfully using custom objects")
        return model
    except Exception as e:
        errors.append(f"Custom objects loading failed: {str(e)}")
        
    # Method 3: Try SavedModel loading
    try:
        model = tf.saved_model.load(model_path)
        logger.info("✅ Model loaded successfully using tf.saved_model.load")
        return model
    except Exception as e:
        errors.append(f"SavedModel loading failed: {str(e)}")
    
    # Method 4: Try loading with h5py directly for HDF5 files
    if model_path.endswith('.h5'):
        try:
            import h5py
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={}
            )
            logger.info("✅ Model loaded successfully using h5py approach")
            return model
        except Exception as e:
            errors.append(f"H5py loading failed: {str(e)}")
    
    # All methods failed
    logger.error(f"❌ All model loading methods failed for {model_path}:")
    for i, error in enumerate(errors):
        logger.error(f"  Method {i+1}: {error}")
    
    return None
