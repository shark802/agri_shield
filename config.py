#!/usr/bin/env python3
"""
Configuration file for AgriShield Flask ML API
Reads from config.php (like PHP files) or environment variables
No hardcoded values - everything is configurable
"""

import os
import re
from pathlib import Path

# ============================================================================
# LOAD FROM config.php (like your PHP files do)
# ============================================================================

def load_config_from_php():
    """Load configuration from config.php file (same as PHP files use)"""
    config = {}
    config_php_path = Path(__file__).resolve().parent.parent / 'config.php'
    
    if config_php_path.exists():
        try:
            with open(config_php_path, 'r') as f:
                content = f.read()
            
            # Extract DB_HOST
            match = re.search(r"define\s*\(\s*['\"]DB_HOST['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_host'] = match.group(1)
            
            # Extract DB_USER
            match = re.search(r"define\s*\(\s*['\"]DB_USER['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_user'] = match.group(1)
            
            # Extract DB_PASS
            match = re.search(r"define\s*\(\s*['\"]DB_PASS['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_password'] = match.group(1)
            
            # Extract DB_NAME
            match = re.search(r"define\s*\(\s*['\"]DB_NAME['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_name'] = match.group(1)
            
        except Exception as e:
            print(f"Warning: Could not read config.php: {e}")
    
    return config

# Load from config.php
php_config = load_config_from_php()

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', '8000'))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# ============================================================================
# DATABASE CONFIGURATION
# Priority: Environment variables > config.php > defaults
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', php_config.get('db_host', 'localhost')),
    'user': os.getenv('DB_USER', php_config.get('db_user', 'root')),
    'password': os.getenv('DB_PASSWORD', php_config.get('db_password', '')),
    'database': os.getenv('DB_NAME', php_config.get('db_name', 'asdb')),
    'charset': os.getenv('DB_CHARSET', 'utf8mb4')
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_BASE_PATH = os.getenv('MODEL_BASE_PATH', str(BASE_DIR))
MODEL_PATHS = [
    Path(MODEL_BASE_PATH) / 'ml_models' / 'pest_detection' / 'best.pt',
    Path(MODEL_BASE_PATH) / 'models' / 'best.pt',
    Path(MODEL_BASE_PATH) / 'datasets' / 'best.pt',
    Path(MODEL_BASE_PATH) / 'datasets' / 'best 2.pt',
    Path(MODEL_BASE_PATH) / 'datasets' / 'best5.pt',
    Path(MODEL_BASE_PATH) / 'best.pt',
    BASE_DIR.parent / 'ml_models' / 'pest_detection' / 'best.pt',
    BASE_DIR.parent / 'pest_detection_ml' / 'models' / 'best.pt',
    BASE_DIR.parent / 'datasets' / 'best 2.pt',
]

# ============================================================================
# DETECTION CONFIGURATION (YOLO Settings)
# ============================================================================

# YOLO Inference Settings
YOLO_IMAGE_SIZE = int(os.getenv('YOLO_IMAGE_SIZE', '640'))
YOLO_BASE_CONFIDENCE = float(os.getenv('YOLO_BASE_CONFIDENCE', '0.15'))
YOLO_IOU_THRESHOLD = float(os.getenv('YOLO_IOU_THRESHOLD', '0.50'))
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'cpu')  # 'cpu' or 'cuda'

# Confidence Thresholds (class-specific)
CONFIDENCE_THRESHOLDS = {
    'Rice_Bug': float(os.getenv('CONF_THRESHOLD_RICE_BUG', '0.20')),
    'black-bug': float(os.getenv('CONF_THRESHOLD_BLACK_BUG', '0.80')),
    'brown_hopper': float(os.getenv('CONF_THRESHOLD_BROWN_HOPPER', '0.15')),
    'green_hopper': float(os.getenv('CONF_THRESHOLD_GREEN_HOPPER', '0.15')),
}
CONFIDENCE_FALLBACK = float(os.getenv('CONF_THRESHOLD_FALLBACK', '0.25'))

# ============================================================================
# FORECASTING CONFIGURATION
# ============================================================================

# Location
DEFAULT_LOCATION = os.getenv('DEFAULT_LOCATION', 'Bago City')

# Risk Calculation Values
RISK_BASE_SCORE = float(os.getenv('RISK_BASE_SCORE', '0.2'))
RISK_TEMP_OPTIMAL = float(os.getenv('RISK_TEMP_OPTIMAL', '0.4'))
RISK_TEMP_NEAR = float(os.getenv('RISK_TEMP_NEAR', '0.2'))
RISK_HUMIDITY_OPTIMAL = float(os.getenv('RISK_HUMIDITY_OPTIMAL', '0.4'))
RISK_HUMIDITY_NEAR = float(os.getenv('RISK_HUMIDITY_NEAR', '0.2'))
RISK_RAINFALL_MODERATE = float(os.getenv('RISK_RAINFALL_MODERATE', '0.1'))
RISK_RAINFALL_HEAVY = float(os.getenv('RISK_RAINFALL_HEAVY', '-0.1'))
RISK_WIND_HIGH = float(os.getenv('RISK_WIND_HIGH', '-0.1'))
RISK_RECENT_DETECTION_BOOST = float(os.getenv('RISK_RECENT_DETECTION_BOOST', '0.2'))

# Risk Level Thresholds
RISK_THRESHOLD_HIGH = float(os.getenv('RISK_THRESHOLD_HIGH', '0.7'))
RISK_THRESHOLD_MEDIUM = float(os.getenv('RISK_THRESHOLD_MEDIUM', '0.4'))

# Database Query Limits
FORECAST_DAYS_BACK = int(os.getenv('FORECAST_DAYS_BACK', '7'))
FORECAST_MAX_DETECTIONS = int(os.getenv('FORECAST_MAX_DETECTIONS', '20'))

# Default Weather Values
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '25'))
DEFAULT_HUMIDITY = float(os.getenv('DEFAULT_HUMIDITY', '70'))
DEFAULT_RAINFALL = float(os.getenv('DEFAULT_RAINFALL', '0'))
DEFAULT_WIND_SPEED = float(os.getenv('DEFAULT_WIND_SPEED', '5'))

# Pest Thresholds (Optimal ranges)
PEST_THRESHOLDS = {
    'rice_bug': {
        'optimal_temp': (
            float(os.getenv('RICE_BUG_TEMP_MIN', '28')),
            float(os.getenv('RICE_BUG_TEMP_MAX', '32'))
        ),
        'optimal_humidity': (
            float(os.getenv('RICE_BUG_HUMIDITY_MIN', '70')),
            float(os.getenv('RICE_BUG_HUMIDITY_MAX', '85'))
        ),
    },
    'green_leaf_hopper': {
        'optimal_temp': (
            float(os.getenv('GREEN_HOPPER_TEMP_MIN', '25')),
            float(os.getenv('GREEN_HOPPER_TEMP_MAX', '30'))
        ),
        'optimal_humidity': (
            float(os.getenv('GREEN_HOPPER_HUMIDITY_MIN', '75')),
            float(os.getenv('GREEN_HOPPER_HUMIDITY_MAX', '90'))
        ),
    },
    'black_bug': {
        'optimal_temp': (
            float(os.getenv('BLACK_BUG_TEMP_MIN', '25')),
            float(os.getenv('BLACK_BUG_TEMP_MAX', '33'))
        ),
        'optimal_humidity': (
            float(os.getenv('BLACK_BUG_HUMIDITY_MIN', '80')),
            float(os.getenv('BLACK_BUG_HUMIDITY_MAX', '95'))
        ),
    },
    'brown_plant_hopper': {
        'optimal_temp': (
            float(os.getenv('BROWN_HOPPER_TEMP_MIN', '24')),
            float(os.getenv('BROWN_HOPPER_TEMP_MAX', '32'))
        ),
        'optimal_humidity': (
            float(os.getenv('BROWN_HOPPER_HUMIDITY_MIN', '75')),
            float(os.getenv('BROWN_HOPPER_HUMIDITY_MAX', '90'))
        ),
    },
}

# Rainfall thresholds for risk calculation
RAINFALL_MODERATE_MIN = float(os.getenv('RAINFALL_MODERATE_MIN', '5'))
RAINFALL_MODERATE_MAX = float(os.getenv('RAINFALL_MODERATE_MAX', '20'))
RAINFALL_HEAVY_THRESHOLD = float(os.getenv('RAINFALL_HEAVY_THRESHOLD', '20'))
WIND_HIGH_THRESHOLD = float(os.getenv('WIND_HIGH_THRESHOLD', '15'))
TEMP_NEAR_OPTIMAL_RANGE = float(os.getenv('TEMP_NEAR_OPTIMAL_RANGE', '5'))
HUMIDITY_NEAR_OPTIMAL_RANGE = float(os.getenv('HUMIDITY_NEAR_OPTIMAL_RANGE', '10'))

# Forecast confidence
FORECAST_CONFIDENCE = float(os.getenv('FORECAST_CONFIDENCE', '0.8'))

# Weather API Configuration (for accurate forecast data)
WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', '76f91b260dc84341a1733851250710')
WEATHERAPI_BASE_URL = os.getenv('WEATHERAPI_BASE_URL', 'http://api.weatherapi.com/v1')
DEFAULT_LATITUDE = float(os.getenv('DEFAULT_LATITUDE', '10.5379'))  # Bago City
DEFAULT_LONGITUDE = float(os.getenv('DEFAULT_LONGITUDE', '122.8386'))  # Bago City

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

