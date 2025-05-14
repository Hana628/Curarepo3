#!/bin/bash

echo "======================================"
echo " Starting Cura Health Assistant..."
echo "======================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python is not installed or not in PATH. Please install Python 3.9 or newer."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "INFO: Found Python version $PYTHON_VERSION"

# Create model_fixed directory if it doesn't exist
if [ ! -d "model_fixed" ]; then
    echo "INFO: Creating model_fixed directory for converted models..."
    mkdir -p model_fixed
fi

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "INFO: Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "INFO: Activating virtual environment..."
source venv/bin/activate

# Install dependencies with exact versions to ensure compatibility
echo "INFO: Installing dependencies with specific versions for compatibility..."
pip install numpy==1.24.3 
pip install h5py==3.9.0 
pip install scikit-learn==1.2.2 
pip install pandas==1.5.3 
pip install pillow==9.5.0
pip install flask==2.2.3 flask-cors==3.0.10 werkzeug==2.2.3 email-validator==1.3.1 gunicorn==20.1.0
pip install joblib==1.2.0 nltk==3.8.1 psycopg2-binary==2.9.6 xgboost==1.7.5 python-dotenv==1.0.0

# Install TensorFlow with specific version
echo "INFO: Installing TensorFlow 2.12.0..."
pip install tensorflow==2.12.0

# Create/update conversion.py script to fix models
echo "INFO: Creating model conversion script..."
cat << 'EOF' > conversion.py
import os
import pickle
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def convert_pickle_model(input_path, output_path):
    """Convert scikit-learn pickle model to be compatible with different scikit-learn versions."""
    try:
        logger.info(f"Attempting to convert model: {input_path} to {output_path}")
        if not os.path.exists(input_path):
            logger.error(f"Input model not found: {input_path}")
            return False

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle module path changes between scikit-learn versions
                if module == 'sklearn.ensemble._forest':
                    module = 'sklearn.ensemble'
                elif module == 'sklearn.tree._classes':
                    module = 'sklearn.tree'
                return super().find_class(module, name)

        # Try to load the model with the custom unpickler
        with open(input_path, 'rb') as f:
            model = CustomUnpickler(f).load()

        # Save it to the new location
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model successfully converted and saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        return False

print("Starting model conversions...")

# Convert disease prediction model
convert_pickle_model("attached_assets/disease_prediction_model.pkl", "model_fixed/disease_prediction_model.pkl")

# Convert blood pressure model
convert_pickle_model("attached_assets/bp_xgb.pkl", "model_fixed/bp_xgb.pkl")

# Convert lifestyle model
convert_pickle_model("attached_assets/lifestyle_model_new.pkl", "model_fixed/lifestyle_model_new.pkl")

print("Model conversion completed.")
EOF

# Create patch script to temporarily modify model paths
echo "INFO: Creating environment setup script..."
cat << 'EOF' > env_setup.py
import sys
import os
print("Setting up environment for local execution...")

# Update model paths if fixed models exist
disease_fixed = os.path.exists("model_fixed/disease_prediction_model.pkl")
bp_fixed = os.path.exists("model_fixed/bp_xgb.pkl")
lifestyle_fixed = os.path.exists("model_fixed/lifestyle_model_new.pkl")

if disease_fixed:
    print("Using fixed disease model")
    os.environ["DISEASE_MODEL_PATH"] = "model_fixed/disease_prediction_model.pkl"
else:
    os.environ["DISEASE_MODEL_PATH"] = "attached_assets/disease_prediction_model.pkl"

if bp_fixed:
    print("Using fixed blood pressure model")
    os.environ["BP_MODEL_PATH"] = "model_fixed/bp_xgb.pkl"
else:
    os.environ["BP_MODEL_PATH"] = "attached_assets/bp_xgb.pkl"

if lifestyle_fixed:
    print("Using fixed lifestyle model")
    os.environ["LIFESTYLE_MODEL_PATH"] = "model_fixed/lifestyle_model_new.pkl"
else:
    os.environ["LIFESTYLE_MODEL_PATH"] = "attached_assets/lifestyle_model_new.pkl"

# Environmental variables to help compatibility
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("Environment setup complete.")
EOF

# Set environment variables for TensorFlow and NumPy compatibility
echo "INFO: Setting environment variables for compatibility..."
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
export TF_CPP_MIN_LOG_LEVEL=3

# Make sure .env file exists
if [ ! -f ".env" ]; then
    echo "INFO: Creating default .env file..."
    echo "SESSION_SECRET=cura_local_development_key" > .env
fi

# Run model conversion
echo "INFO: Converting models for local compatibility..."
python3 conversion.py
if [ $? -ne 0 ]; then
    echo "WARNING: Model conversion had issues but will continue."
fi

# Check model files
echo "INFO: Verifying required model files..."
if [ ! -f "attached_assets/disease_prediction_model.pkl" ]; then
    echo "WARNING: Disease prediction model not found in attached_assets folder."
fi
if [ ! -f "attached_assets/model (1).h5" ]; then
    echo "WARNING: Skin disease model not found in attached_assets folder."
fi

# Start the application
echo
echo "======================================"
echo " Starting Cura Health Assistant"
echo "======================================"
echo
echo "INFO: Application will be available at: http://localhost:5000"
echo
echo "INFO: Press Ctrl+C to stop the application"
echo
python3 -c "import env_setup" && python3 main.py