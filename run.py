#!/usr/bin/env python3
"""
Startup script for the NASA Weather Prediction API
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import HOST, PORT, DEBUG, LOG_LEVEL

def main():
    """Main function to start the FastAPI server"""
    print("🌍 Starting NASA Weather Prediction API...")
    print(f"📍 Server will be available at: http://{HOST}:{PORT}")
    print(f"📚 API Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔄 Debug Mode: {DEBUG}")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()