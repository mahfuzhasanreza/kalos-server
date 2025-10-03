import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# API Configuration
API_TITLE = "NASA Weather Prediction API"
API_DESCRIPTION = "Global Weather Prediction & Recommendation System using NASA data and AI models"
API_VERSION = "1.0.0"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "*"  # Remove this in production
]

# NASA Data Configuration
NASA_GEOS_URL = "http://opendap.nccs.nasa.gov:80/dods/gmao/geos-cf/fcast/met_inst_1hr_g1440x721_p23.latest"
NASA_GPM_URL = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"

# Model Configuration
MODEL_TRAINING_SAMPLES = 15000
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.2

# Random Forest Parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH_TEMP = 20
RF_MAX_DEPTH_OTHER = 15
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

# Forecast Configuration
MAX_FORECAST_DAYS = 30
MAX_FUTURE_PREDICTION_DAYS = 180  # 6 months

# Weather Bounds
TEMP_MIN = -40
TEMP_MAX = 50
HUMIDITY_MIN = 10
HUMIDITY_MAX = 100
WIND_SPEED_MAX = 50
PRECIP_PROB_MAX = 100

# Default Values
DEFAULT_PRESSURE = 1013.25
DEFAULT_ELEVATION = 0
DEFAULT_CONFIDENCE = 85

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"