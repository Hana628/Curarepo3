# Cura Health Assistant - Desktop Installation Guide

This guide explains how to run Cura Health Assistant on your local desktop machine.

## Windows Installation

### First-time Setup

1. Make sure you have Python 3.9 or newer installed
2. Double-click `setup.bat` to install all required dependencies
   - This will create a virtual environment with all necessary packages
   - You only need to do this once

### Running the Application

1. Double-click `run.bat` to start the application
2. Open your web browser and go to http://localhost:5000
3. To stop the application, press Ctrl+C in the command window

## Linux/macOS Installation

### First-time Setup

1. Make sure you have Python 3.9 or newer installed
2. Make the setup script executable: `chmod +x setup.sh`
3. Run the setup script: `./setup.sh`
   - This will create a virtual environment with all necessary packages
   - You only need to do this once

### Running the Application

1. Make the run script executable: `chmod +x run.sh`
2. Start the application: `./run.sh`
3. Open your web browser and go to http://localhost:5000
4. To stop the application, press Ctrl+C in the terminal

## Troubleshooting

### Microsoft Defender Warning

If Microsoft Defender SmartScreen shows a warning when you run the batch files:

1. Click "More info" on the warning dialog
2. Click "Run anyway"

This happens because the batch files are not digitally signed. The files are safe to run.

### Dependencies Issues

If you encounter issues with TensorFlow or other dependencies:

1. Try running `pip install tensorflow==2.12.0` manually
2. If that fails, the application will still work using fallback prediction methods

### Model Files

Make sure the following files exist in the `attached_assets` folder:
- disease_prediction_model.pkl
- bp_xgb.pkl 
- lifestyle_model_new.pkl
- ecg_lstm_model.h5
- model (1).h5
- trained_model.h5