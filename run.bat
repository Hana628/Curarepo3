@echo off
echo ======================================
echo  Starting Cura Health Assistant...
echo ======================================
echo.

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH. Please install Python 3.9 or newer.
    goto :error
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo INFO: Found Python version %PYTHON_VERSION%

:: Create model_fixed directory if it doesn't exist
if not exist model_fixed\ (
    echo INFO: Creating model_fixed directory for converted models...
    mkdir model_fixed
)

:: Check if venv exists, create if not
if not exist venv\ (
    echo INFO: Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        goto :error
    )
)

:: Activate virtual environment
echo INFO: Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies with exact versions to ensure compatibility
echo INFO: Installing dependencies with specific versions for compatibility...
pip install numpy==1.24.3 
pip install h5py==3.9.0 
pip install scikit-learn==1.2.2 
pip install pandas==1.5.3 
pip install pillow==9.5.0
pip install flask==2.2.3 flask-cors==3.0.10 werkzeug==2.2.3 email-validator==1.3.1 gunicorn==20.1.0
pip install joblib==1.2.0 nltk==3.8.1 psycopg2-binary==2.9.6 xgboost==1.7.5 python-dotenv==1.0.0

:: Install TensorFlow with specific version
echo INFO: Installing TensorFlow 2.12.0...
pip install tensorflow==2.12.0

:: Create/update conversion.py script to fix models
echo INFO: Creating model conversion script...
echo import os > conversion.py
echo import pickle >> conversion.py
echo import numpy as np >> conversion.py
echo import sys >> conversion.py
echo import logging >> conversion.py
echo. >> conversion.py
echo logging.basicConfig(level=logging.INFO) >> conversion.py
echo logger = logging.getLogger() >> conversion.py
echo. >> conversion.py
echo def convert_pickle_model(input_path, output_path): >> conversion.py
echo     """Convert scikit-learn pickle model to be compatible with different scikit-learn versions.""" >> conversion.py
echo     try: >> conversion.py
echo         logger.info(f"Attempting to convert model: {input_path} to {output_path}") >> conversion.py
echo         if not os.path.exists(input_path): >> conversion.py
echo             logger.error(f"Input model not found: {input_path}") >> conversion.py
echo             return False >> conversion.py
echo. >> conversion.py
echo         class CustomUnpickler(pickle.Unpickler): >> conversion.py
echo             def find_class(self, module, name): >> conversion.py
echo                 # Handle module path changes between scikit-learn versions >> conversion.py
echo                 if module == 'sklearn.ensemble._forest': >> conversion.py
echo                     module = 'sklearn.ensemble' >> conversion.py
echo                 elif module == 'sklearn.tree._classes': >> conversion.py
echo                     module = 'sklearn.tree' >> conversion.py
echo                 return super().find_class(module, name) >> conversion.py
echo. >> conversion.py
echo         # Try to load the model with the custom unpickler >> conversion.py
echo         with open(input_path, 'rb') as f: >> conversion.py
echo             model = CustomUnpickler(f).load() >> conversion.py
echo. >> conversion.py
echo         # Save it to the new location >> conversion.py
echo         with open(output_path, 'wb') as f: >> conversion.py
echo             pickle.dump(model, f) >> conversion.py
echo. >> conversion.py
echo         logger.info(f"Model successfully converted and saved to {output_path}") >> conversion.py
echo         return True >> conversion.py
echo     except Exception as e: >> conversion.py
echo         logger.error(f"Error converting model: {str(e)}") >> conversion.py
echo         return False >> conversion.py
echo. >> conversion.py
echo print("Starting model conversions...") >> conversion.py
echo. >> conversion.py
echo # Convert disease prediction model >> conversion.py
echo convert_pickle_model("attached_assets/disease_prediction_model.pkl", "model_fixed/disease_prediction_model.pkl") >> conversion.py
echo. >> conversion.py
echo # Convert blood pressure model >> conversion.py
echo convert_pickle_model("attached_assets/bp_xgb.pkl", "model_fixed/bp_xgb.pkl") >> conversion.py
echo. >> conversion.py
echo # Convert lifestyle model >> conversion.py
echo convert_pickle_model("attached_assets/lifestyle_model_new.pkl", "model_fixed/lifestyle_model_new.pkl") >> conversion.py
echo. >> conversion.py
echo print("Model conversion completed.") >> conversion.py

:: Set environment variables for TensorFlow and NumPy compatibility
echo INFO: Setting environment variables for compatibility...
set NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
set TF_CPP_MIN_LOG_LEVEL=3

:: Create patch script to temporarily modify model paths
echo INFO: Creating environment setup script...
echo import sys > env_setup.py
echo import os >> env_setup.py
echo print("Setting up environment for local execution...") >> env_setup.py
echo. >> env_setup.py
echo # Update model paths if fixed models exist >> env_setup.py
echo disease_fixed = os.path.exists("model_fixed/disease_prediction_model.pkl") >> env_setup.py
echo bp_fixed = os.path.exists("model_fixed/bp_xgb.pkl") >> env_setup.py
echo lifestyle_fixed = os.path.exists("model_fixed/lifestyle_model_new.pkl") >> env_setup.py
echo. >> env_setup.py
echo if disease_fixed: >> env_setup.py
echo     print("Using fixed disease model") >> env_setup.py
echo     os.environ["DISEASE_MODEL_PATH"] = "model_fixed/disease_prediction_model.pkl" >> env_setup.py
echo else: >> env_setup.py
echo     os.environ["DISEASE_MODEL_PATH"] = "attached_assets/disease_prediction_model.pkl" >> env_setup.py
echo. >> env_setup.py
echo if bp_fixed: >> env_setup.py
echo     print("Using fixed blood pressure model") >> env_setup.py
echo     os.environ["BP_MODEL_PATH"] = "model_fixed/bp_xgb.pkl" >> env_setup.py
echo else: >> env_setup.py
echo     os.environ["BP_MODEL_PATH"] = "attached_assets/bp_xgb.pkl" >> env_setup.py
echo. >> env_setup.py
echo if lifestyle_fixed: >> env_setup.py
echo     print("Using fixed lifestyle model") >> env_setup.py
echo     os.environ["LIFESTYLE_MODEL_PATH"] = "model_fixed/lifestyle_model_new.pkl" >> env_setup.py
echo else: >> env_setup.py
echo     os.environ["LIFESTYLE_MODEL_PATH"] = "attached_assets/lifestyle_model_new.pkl" >> env_setup.py
echo. >> env_setup.py
echo # Environmental variables to help compatibility >> env_setup.py
echo os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0" >> env_setup.py
echo os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" >> env_setup.py
echo print("Environment setup complete.") >> env_setup.py

:: Make sure .env file exists
if not exist .env (
    echo INFO: Creating default .env file...
    echo SESSION_SECRET=cura_local_development_key > .env
)

:: Run model conversion
echo INFO: Converting models for local compatibility...
python conversion.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Model conversion had issues but will continue.
)

:: Check model files
echo INFO: Verifying required model files...
if not exist attached_assets\disease_prediction_model.pkl (
    echo WARNING: Disease prediction model not found in attached_assets folder.
)
if not exist "attached_assets\model (1).h5" (
    echo WARNING: Skin disease model not found in attached_assets folder.
)

:: Start the application
echo.
echo ======================================
echo  Starting Cura Health Assistant
echo ======================================
echo.
echo INFO: Application will be available at: http://localhost:5000
echo.
echo INFO: Press Ctrl+C to stop the application
echo.
python -c "import env_setup" && python main.py

goto :end

:error
echo.
echo ERROR: An error occurred. Please check the messages above.
pause
exit /b 1

:end
echo.
echo INFO: Cura Health Assistant has stopped.
pause