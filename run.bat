@echo off
echo Starting Cura Health Assistant...

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9 or newer.
    goto :error
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version %PYTHON_VERSION%

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

:: Install dependencies if requirements_github.txt exists
if exist requirements_github.txt (
    echo Installing dependencies from requirements_github.txt...
    pip install -r requirements_github.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Some packages could not be installed. Trying to install core packages manually...
        pip install flask flask-cors werkzeug email-validator gunicorn
    )
) else (
    echo requirements_github.txt not found. Installing core packages...
    pip install flask flask-cors werkzeug email-validator gunicorn
)

:: Make sure .env file exists
if not exist .env (
    echo Creating default .env file...
    echo SESSION_SECRET=cura_local_development_key > .env
)

:: Start the application
echo Starting the application...
echo Access the application at http://localhost:5000
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