@echo off
echo Setting up Cura Health Assistant...

:: Check Python installation
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.9 or newer.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. 
    echo You may need to install the virtualenv package: pip install virtualenv
    pause
    exit /b 1
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

:: Create .env file
echo Creating environment file...
echo SESSION_SECRET=cura_local_development_key > .env

echo.
echo Setup complete! Run "run.bat" to start the application.
pause