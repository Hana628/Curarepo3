#!/bin/bash

echo "Setting up Cura Health Assistant..."

# Check Python installation
if ! command -v python3 &> /dev/null
then
    echo "Python 3 not found. Please install Python 3.9 or newer."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    echo "You may need to install venv: sudo apt-get install python3-venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_github.txt
if [ $? -ne 0 ]; then
    echo "Some packages could not be installed. Installing core packages..."
    pip install flask==2.2.3 flask-cors==3.0.10 gunicorn==20.1.0
    pip install scikit-learn==1.2.2 pandas==1.5.3 numpy==1.24.3 
    pip install joblib==1.2.0 pillow==9.5.0 python-dotenv==1.0.0
fi

# Create .env file
echo "Creating environment file..."
echo "SESSION_SECRET=cura_local_development_key" > .env

echo ""
echo "Setup complete! Run \"./run.sh\" to start the application."