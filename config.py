#!/usr/bin/env python3
"""
Production Configuration for AgriShield ML Flask API
Loads settings from config.php for database credentials
"""

import os
import re
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask settings
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Production mode (disable debug)
if os.getenv('PRODUCTION', 'false').lower() == 'true':
    FLASK_DEBUG = False

# Load database config from config.php
def load_db_config_from_php():
    """Load database configuration from config.php"""
    config_php_path = BASE_DIR / 'config.php'
    
    if not config_php_path.exists():
        # Fallback to environment variables
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'asdb'),
            'charset': os.getenv('DB_CHARSET', 'utf8mb4')
        }
    
    try:
        with open(config_php_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract database settings from PHP defines
        config = {}
        
        # Extract DB_HOST
        match = re.search(r"define\s*\(\s*['\"]DB_HOST['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
        if match:
            config['host'] = match.group(1)
        else:
            config['host'] = os.getenv('DB_HOST', 'localhost')
        
        # Extract DB_USER
        match = re.search(r"define\s*\(\s*['\"]DB_USER['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
        if match:
            config['user'] = match.group(1)
        else:
            config['user'] = os.getenv('DB_USER', 'root')
        
        # Extract DB_PASS
        match = re.search(r"define\s*\(\s*['\"]DB_PASS['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
        if match:
            config['password'] = match.group(1)
        else:
            config['password'] = os.getenv('DB_PASSWORD', '')
        
        # Extract DB_NAME
        match = re.search(r"define\s*\(\s*['\"]DB_NAME['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
        if match:
            config['database'] = match.group(1)
        else:
            config['database'] = os.getenv('DB_NAME', 'asdb')
        
        config['charset'] = os.getenv('DB_CHARSET', 'utf8mb4')
        
        return config
    except Exception as e:
        print(f"Warning: Could not load config.php: {e}")
        # Fallback to environment variables
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'asdb'),
            'charset': os.getenv('DB_CHARSET', 'utf8mb4')
        }

# Database configuration (from config.php)
DB_CONFIG = load_db_config_from_php()

# Get PHP base URL for API calls (production-ready)
def get_php_base_url():
    """Get PHP base URL for API calls"""
    # Check environment variable first
    php_url = os.getenv('PHP_BASE_URL')
    if php_url:
        return php_url.rstrip('/')
    
    # Try to detect from common production setups
    # This should be set via environment variable in production
    # Default to Heroku URL if on Heroku, otherwise localhost
    if os.getenv('DYNO'):  # DYNO is set on Heroku
        return 'https://endpoint-agrishield-12fa38de0135.herokuapp.com'
    return os.getenv('PHP_BASE_URL', 'http://localhost/Proto1')

PHP_BASE_URL = get_php_base_url()

# Model paths (priority order)
MODEL_PATHS = [
    BASE_DIR / 'ml_models' / 'pest_detection' / 'best.pt',
    BASE_DIR / 'models' / 'best.pt',
    BASE_DIR / 'datasets' / 'best.pt',
    BASE_DIR / 'datasets' / 'best 2.pt',
    BASE_DIR / 'datasets' / 'best5.pt',
    BASE_DIR / 'best.pt',
    BASE_DIR / 'pest_detection_ml' / 'models' / 'best.pt',
    BASE_DIR / 'datasets' / 'best 2.pt',
]

# Default location
DEFAULT_LOCATION = os.getenv('DEFAULT_LOCATION', 'Bago City')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# YOLO Detection settings
YOLO_IMAGE_SIZE = int(os.getenv('YOLO_IMAGE_SIZE', '640'))
YOLO_BASE_CONFIDENCE = float(os.getenv('YOLO_BASE_CONFIDENCE', '0.15'))
YOLO_IOU_THRESHOLD = float(os.getenv('YOLO_IOU_THRESHOLD', '0.50'))
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'cpu')

# Confidence thresholds per pest
CONFIDENCE_THRESHOLDS = {
    'Rice_Bug': float(os.getenv('CONF_RICE_BUG', '0.20')),
    'black-bug': float(os.getenv('CONF_BLACK_BUG', '0.80')),
    'brown_hopper': float(os.getenv('CONF_BROWN_HOPPER', '0.15')),
    'green_hopper': float(os.getenv('CONF_GREEN_HOPPER', '0.15')),
}
CONFIDENCE_FALLBACK = float(os.getenv('CONFIDENCE_FALLBACK', '0.25'))

# Forecasting settings
RISK_BASE_SCORE = float(os.getenv('RISK_BASE_SCORE', '0.2'))
RISK_TEMP_OPTIMAL = float(os.getenv('RISK_TEMP_OPTIMAL', '0.4'))
RISK_TEMP_NEAR = float(os.getenv('RISK_TEMP_NEAR', '0.2'))
RISK_HUMIDITY_OPTIMAL = float(os.getenv('RISK_HUMIDITY_OPTIMAL', '0.4'))
RISK_HUMIDITY_NEAR = float(os.getenv('RISK_HUMIDITY_NEAR', '0.2'))
RISK_RAINFALL_MODERATE = float(os.getenv('RISK_RAINFALL_MODERATE', '0.1'))
RISK_RAINFALL_HEAVY = float(os.getenv('RISK_RAINFALL_HEAVY', '-0.1'))
RISK_WIND_HIGH = float(os.getenv('RISK_WIND_HIGH', '-0.1'))
RISK_RECENT_DETECTION_BOOST = float(os.getenv('RISK_RECENT_DETECTION_BOOST', '0.2'))
RISK_THRESHOLD_HIGH = float(os.getenv('RISK_THRESHOLD_HIGH', '0.7'))
RISK_THRESHOLD_MEDIUM = float(os.getenv('RISK_THRESHOLD_MEDIUM', '0.4'))
FORECAST_DAYS_BACK = int(os.getenv('FORECAST_DAYS_BACK', '7'))
FORECAST_MAX_DETECTIONS = int(os.getenv('FORECAST_MAX_DETECTIONS', '20'))
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '25'))
DEFAULT_HUMIDITY = float(os.getenv('DEFAULT_HUMIDITY', '70'))
DEFAULT_RAINFALL = float(os.getenv('DEFAULT_RAINFALL', '0'))
DEFAULT_WIND_SPEED = float(os.getenv('DEFAULT_WIND_SPEED', '5'))

# Pest thresholds
PEST_THRESHOLDS = {
    'rice_bug': {'optimal_temp': (28, 32), 'optimal_humidity': (70, 85)},
    'black-bug': {'optimal_temp': (25, 30), 'optimal_humidity': (75, 90)},
    'brown_hopper': {'optimal_temp': (26, 32), 'optimal_humidity': (80, 95)},
    'green_hopper': {'optimal_temp': (25, 30), 'optimal_humidity': (70, 85)},
}

RAINFALL_MODERATE_MIN = float(os.getenv('RAINFALL_MODERATE_MIN', '5'))
RAINFALL_MODERATE_MAX = float(os.getenv('RAINFALL_MODERATE_MAX', '20'))
RAINFALL_HEAVY_THRESHOLD = float(os.getenv('RAINFALL_HEAVY_THRESHOLD', '20'))
WIND_HIGH_THRESHOLD = float(os.getenv('WIND_HIGH_THRESHOLD', '15'))
TEMP_NEAR_OPTIMAL_RANGE = float(os.getenv('TEMP_NEAR_OPTIMAL_RANGE', '3'))
HUMIDITY_NEAR_OPTIMAL_RANGE = float(os.getenv('HUMIDITY_NEAR_OPTIMAL_RANGE', '10'))
FORECAST_CONFIDENCE = float(os.getenv('FORECAST_CONFIDENCE', '0.7'))
DEFAULT_LATITUDE = float(os.getenv('DEFAULT_LATITUDE', '10.5388'))
DEFAULT_LONGITUDE = float(os.getenv('DEFAULT_LONGITUDE', '122.8383'))
