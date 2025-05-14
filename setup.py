#!/usr/bin/env python
"""
Setup script for CURA Health Assistant.
This is not a traditional Python package setup script, but rather a helper
to set up the project environment when cloned from GitHub.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('setup')

def check_environment():
    """Check if we have all the necessary tools installed."""
    logger.info("Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or newer is required!")
        return False
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if pip is available
    try:
        import pip
        logger.info(f"pip version: {pip.__version__}")
    except ImportError:
        logger.error("pip is not installed!")
        return False
    
    return True

def setup_env_file():
    """Set up the .env file if it doesn't exist."""
    env_file = Path('.env')
    example_file = Path('.env.example')
    
    if env_file.exists():
        logger.info(".env file already exists, skipping")
        return
    
    if not example_file.exists():
        logger.error(".env.example file not found!")
        return
    
    # Copy the example file
    shutil.copy(example_file, env_file)
    logger.info("Created .env file from .env.example")
    logger.info("Please edit the .env file and set your API keys and other configuration")

def main():
    """Main setup function."""
    logger.info("Setting up CURA Health Assistant...")
    
    if not check_environment():
        return
    
    setup_env_file()
    
    logger.info("")
    logger.info("Setup completed!")
    logger.info("Next steps:")
    logger.info("1. Edit the .env file to set your API keys")
    logger.info("2. Install dependencies: pip install -r requirements_github.txt")
    logger.info("3. Run the app: python main.py")

if __name__ == '__main__':
    main()