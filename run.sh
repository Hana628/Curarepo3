#!/bin/bash

echo "Starting Cura Health Assistant..."

# Check Python installation
if ! command -v python3 &> /dev/null
then
    echo "Python not found. Please install Python 3.9 or newer."
    exit 1
fi

# Setup environment
if [ ! -f ".env" ]; then
    echo "SESSION_SECRET=cura_local_development_key" > .env
fi

# Set environment variables
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
export TF_CPP_MIN_LOG_LEVEL=3

# Start in virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Using system Python installation..."
fi

# Start the application
echo "Starting the application at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python3 main.py