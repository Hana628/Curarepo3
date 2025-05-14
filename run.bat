@echo off
echo Starting Cura Health Assistant...

:: Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.9 or newer is required but not found in PATH.
    goto :error
)

:: Check if venv exists, create if not
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment.
        goto :error
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements_github.txt
if %ERRORLEVEL% NEQ 0 (
    echo Some packages could not be installed. Installing core packages...
    pip install flask==2.2.3 flask-cors==3.0.10 gunicorn==20.1.0
    pip install scikit-learn==1.2.2 pandas==1.5.3 numpy==1.24.3 
    pip install joblib==1.2.0 pillow==9.5.0 python-dotenv==1.0.0
)

:: Set environment variables for compatibility
echo Setting environment variables for compatibility...
set NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
set TF_CPP_MIN_LOG_LEVEL=3
set DISEASE_MODEL_PATH=attached_assets\disease_prediction_model.pkl
set BP_MODEL_PATH=attached_assets\bp_xgb.pkl
set LIFESTYLE_MODEL_PATH=attached_assets\lifestyle_model_new.pkl
set ANOMALY_MODEL_PATH=attached_assets\ecg_lstm_model.h5
set SKIN_MODEL_PATH=attached_assets\model (1).h5
set DIABETES_MODEL_PATH=attached_assets\trained_model.h5

:: Create .env file if needed
if not exist .env (
    echo Creating default .env file...
    echo SESSION_SECRET=cura_local_development_key > .env
)

:: Start the application
echo Starting the application...
echo The application will be available at: http://localhost:5000
python main.py

goto :end

:error
echo.
echo An error occurred. Please check the error messages above.
pause
exit /b 1

:end
echo.
echo Cura Health Assistant has stopped.
pause