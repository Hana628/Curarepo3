# Troubleshooting Guide for Cura Health Assistant

This guide contains solutions for common issues you might encounter when running Cura Health Assistant locally.

## Installation Issues

### Python Version Issues

**Issue**: Error about incompatible Python version.

**Solution**:
- Make sure you have Python 3.9 or newer installed.
- Some packages like TensorFlow might have specific version requirements.
- Try creating a fresh virtual environment: `python -m venv fresh_venv`

### Package Installation Fails

**Issue**: Errors when installing packages from requirements_github.txt.

**Solution**:
- Install core packages separately: `pip install flask flask-cors gunicorn`
- For numpy/TensorFlow issues, try: `pip install numpy==1.24.3 tensorflow==2.12.0`
- For Windows, you might need to install some packages from wheels (.whl files) if you encounter compilation issues.

### Missing Model Files

**Issue**: Errors about missing model files.

**Solution**:
- Make sure all model files are in the `attached_assets` directory.
- Required files: `bp_xgb.pkl`, `diabetes_model.keras`, `disease_prediction_model.pkl`, `lifestyle_recommendation_model.pkl`, `lstm_anomaly_model.keras`
- If any file is missing, you'll need to download them from the original repository or contact the repository owner.

## Running Issues

### Application Won't Start

**Issue**: Error when trying to run the application.

**Solution**:
- Check if port 5000 is already in use. Try changing to another port (e.g., 5001) or kill the process using port 5000.
- Check if you're using the correct Python version within your virtual environment.
- Try running with the command: `flask run --host=0.0.0.0 --port=5000`
- If using gunicorn, try: `gunicorn --bind 0.0.0.0:5000 main:app`

### ModuleNotFoundError

**Issue**: Python says it can't find a module.

**Solution**:
- Make sure you've activated your virtual environment.
- Install the missing module: `pip install [module_name]`
- For custom modules, check file paths and imports.

### Permission Errors

**Issue**: Permission denied when running scripts or accessing files.

**Solution**:
- For Linux/macOS, make scripts executable: `chmod +x run.sh`
- Run terminal as administrator (Windows) or use sudo (Linux/macOS).
- Check file and directory permissions.

## Usage Issues

### Model Prediction Failures

**Issue**: Errors when trying to use prediction features.

**Solution**:
- Check the model loading logs in the console/terminal.
- Verify that input data matches expected format.
- For models that fail, the application has fallback mechanisms.

### Browser Can't Connect

**Issue**: Browser shows "Connection refused" or similar error.

**Solution**:
- Make sure you're using the correct URL (http://localhost:5000).
- Check that the application is still running in the terminal.
- Try using 127.0.0.1 instead of localhost.
- Check if any firewall or security software is blocking connections.

### Slow Performance

**Issue**: Application runs slowly.

**Solution**:
- TensorFlow models can be resource-intensive. Ensure your computer meets the minimum requirements.
- Close other resource-intensive applications.
- Consider setting `TF_FORCE_GPU_ALLOW_GROWTH=true` environment variable if using GPU.

## Database Issues

**Issue**: Database-related errors.

**Solution**:
- Cura typically uses SQLite which doesn't require additional setup.
- Check if the database file is created and writable.
- If using PostgreSQL, make sure it's properly installed and configured.

## Other Issues

### JavaScript Console Errors

**Issue**: Frontend features don't work; console shows errors.

**Solution**:
- Press F12 in your browser to open developer tools.
- Check the console for specific error messages.
- Sometimes clearing browser cache or doing a hard refresh (Ctrl+F5) can help.

### Missing Icons or Styling

**Issue**: Website appears broken or unstyled.

**Solution**:
- Make sure you have an internet connection for loading Bootstrap and FontAwesome.
- Check the browser's network tab (F12) for failed resource loads.

## Still Having Issues?

If the above solutions don't help, try:

1. Check the application logs for specific error messages.
2. Search online for the specific error message.
3. Restart your computer and try again.
4. Contact the original repository owner for assistance with model files or code-specific issues.