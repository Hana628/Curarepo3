@echo off
echo Starting Cura Health Assistant...

:: Check Python installation
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.9 or newer.
    pause
    exit /b 1
)

:: Setup environment
if not exist .env (
    echo SESSION_SECRET=cura_local_development_key > .env
)

:: Set environment variables 
set NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
set TF_CPP_MIN_LOG_LEVEL=3

:: Start in virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Using system Python installation...
)

:: Start the application
echo Starting the application at http://localhost:5000
echo Press Ctrl+C to stop the server
python main.py

pause