# Local Installation Guide for Cura Health Assistant

This guide will help you run Cura Health Assistant locally on your computer without using GitHub.

## Prerequisites

1. **Python 3.9+**: Make sure you have Python 3.9 or newer installed on your system.
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify your installation by running: `python --version` or `python3 --version`

2. **Pip**: Python's package installer (usually comes with Python).
   - Verify with: `pip --version` or `pip3 --version`

## Step 1: Download the Application

1. Download the application files from Replit.
   - Either use the "Download as zip" option in Replit, or
   - Copy all files to your local machine manually

2. Extract/place all files in a directory of your choice, such as:
   - Windows: `C:\Users\YourName\Cura`
   - macOS/Linux: `/Users/YourName/Cura` or `/home/YourName/Cura`

## Step 2: Setup a Virtual Environment (Optional but Recommended)

Creating a virtual environment keeps dependencies for different projects separate.

### Windows
```
cd C:\path\to\Cura
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux
```
cd /path/to/Cura
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Required Packages

Install all necessary dependencies:

```
pip install -r requirements_github.txt
```

If that doesn't work, install the packages explicitly:

```
pip install flask flask-cors gunicorn email-validator flask-sqlalchemy joblib nltk numpy pandas pillow psycopg2-binary scikit-learn tensorflow werkzeug xgboost
```

## Step 4: Create a .env File

Create a file named `.env` in the root directory with the following content:

```
SESSION_SECRET=your_secret_key_here
```

Note: Replace `your_secret_key_here` with any random string for security.

## Step 5: Prepare the Application

Before running the app, make sure all the model files are in the `attached_assets` directory:
- bp_xgb.pkl
- diabetes_model.keras
- disease_prediction_model.pkl
- lifestyle_recommendation_model.pkl
- lstm_anomaly_model.keras

## Step 6: Run the Application

### Windows
```
python main.py
```
or
```
flask run --host=0.0.0.0 --port=5000
```

### macOS/Linux
```
python3 main.py
```
or
```
flask run --host=0.0.0.0 --port=5000
```

If these commands don't work, try:
```
gunicorn --bind 0.0.0.0:5000 main:app
```

## Step 7: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

You should now see the Cura Health Assistant running locally on your computer!

## Troubleshooting

1. **Port already in use**: If port 5000 is already in use, change the port number:
   ```
   flask run --host=0.0.0.0 --port=5001
   ```
   Then access the app at `http://localhost:5001`

2. **Missing dependencies**: If you see errors about missing packages, install them with:
   ```
   pip install package_name
   ```

3. **Model loading issues**: If the models fail to load, check that all files are in the correct locations and formats.

4. **Permission errors**: On Linux/macOS, you might need to use `sudo` or adjust file permissions.

5. **Python version conflicts**: Make sure you're using Python 3.9 or newer.

## Running on Startup (Optional)

### Windows
Create a batch file (run_cura.bat) with:
```
@echo off
cd C:\path\to\Cura
call venv\Scripts\activate.bat
python main.py
```

### macOS/Linux
Create a shell script (run_cura.sh) with:
```
#!/bin/bash
cd /path/to/Cura
source venv/bin/activate
python3 main.py
```
Make it executable: `chmod +x run_cura.sh`

## Notes

- This is a standalone version that doesn't require GitHub integration
- All prediction models run locally for privacy
- No internet connection required after installation
- All health data stays on your computer

For advanced configuration options, see the `README.md` file.