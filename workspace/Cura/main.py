import os
from app import app  # noqa: F401
import logging

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the application
    print(f"Starting CURA Health Assistant on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
