import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

def log_request(method: str, endpoint: str, params: dict = None):
    """Log API requests"""
    logger = logging.getLogger("api")
    message = f"{method} {endpoint}"
    if params:
        message += f" - Params: {params}"
    logger.info(message)

def log_error(error: Exception, context: str = ""):
    """Log errors consistently"""
    logger = logging.getLogger("error")
    message = f"Error in {context}: {str(error)}"
    logger.error(message, exc_info=True)

def log_weather_fetch(source: str, location: str, success: bool):
    """Log weather data fetch attempts"""
    logger = logging.getLogger("weather")
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Weather fetch {status} - Source: {source}, Location: {location}")

def log_model_training(model_name: str, metrics: dict = None):
    """Log model training results"""
    logger = logging.getLogger("model")
    message = f"Model trained: {model_name}"
    if metrics:
        message += f" - Metrics: {metrics}"
    logger.info(message)