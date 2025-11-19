#!/usr/bin/env python3
"""
AgriShield ML Flask API
Converted from Django to Flask for easier deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
import os
import time
import io
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
import threading
import time

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Load configuration
try:
    from config import (
        FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
        DB_CONFIG, MODEL_PATHS, DEFAULT_LOCATION, LOG_LEVEL,
        # Detection settings
        YOLO_IMAGE_SIZE, YOLO_BASE_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_DEVICE,
        CONFIDENCE_THRESHOLDS, CONFIDENCE_FALLBACK,
        # Forecasting settings
        RISK_BASE_SCORE, RISK_TEMP_OPTIMAL, RISK_TEMP_NEAR,
        RISK_HUMIDITY_OPTIMAL, RISK_HUMIDITY_NEAR,
        RISK_RAINFALL_MODERATE, RISK_RAINFALL_HEAVY, RISK_WIND_HIGH,
        RISK_RECENT_DETECTION_BOOST, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM,
        FORECAST_DAYS_BACK, FORECAST_MAX_DETECTIONS,
        DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY, DEFAULT_RAINFALL, DEFAULT_WIND_SPEED,
        PEST_THRESHOLDS,
        RAINFALL_MODERATE_MIN, RAINFALL_MODERATE_MAX, RAINFALL_HEAVY_THRESHOLD,
        WIND_HIGH_THRESHOLD, TEMP_NEAR_OPTIMAL_RANGE, HUMIDITY_NEAR_OPTIMAL_RANGE,
        FORECAST_CONFIDENCE, DEFAULT_LATITUDE, DEFAULT_LONGITUDE
    )
    USE_CONFIG_FILE = True
except ImportError:
    # Fallback if config.py doesn't exist
    USE_CONFIG_FILE = False
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '8000'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'asdb'),
        'charset': os.getenv('DB_CHARSET', 'utf8mb4')
    }
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATHS = [
        BASE_DIR / 'ml_models' / 'pest_detection' / 'best.pt',
        BASE_DIR / 'models' / 'best.pt',
        BASE_DIR / 'datasets' / 'best.pt',
        BASE_DIR / 'datasets' / 'best 2.pt',
        BASE_DIR / 'datasets' / 'best5.pt',
        BASE_DIR / 'best.pt',
        BASE_DIR.parent / 'ml_models' / 'pest_detection' / 'best.pt',
        BASE_DIR.parent / 'pest_detection_ml' / 'models' / 'best.pt',
        BASE_DIR.parent / 'datasets' / 'best 2.pt',
    ]
    DEFAULT_LOCATION = os.getenv('DEFAULT_LOCATION', 'Bago City')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    # Detection defaults
    YOLO_IMAGE_SIZE = 640
    YOLO_BASE_CONFIDENCE = 0.15
    YOLO_IOU_THRESHOLD = 0.50
    YOLO_DEVICE = 'cpu'
    CONFIDENCE_THRESHOLDS = {
        'Rice_Bug': 0.20,
        'black-bug': 0.80,
        'brown_hopper': 0.15,
        'green_hopper': 0.15,
    }
    CONFIDENCE_FALLBACK = 0.25
    # Forecasting defaults
    RISK_BASE_SCORE = 0.2
    RISK_TEMP_OPTIMAL = 0.4
    RISK_TEMP_NEAR = 0.2
    RISK_HUMIDITY_OPTIMAL = 0.4
    RISK_HUMIDITY_NEAR = 0.2
    RISK_RAINFALL_MODERATE = 0.1
    RISK_RAINFALL_HEAVY = -0.1
    RISK_WIND_HIGH = -0.1
    RISK_RECENT_DETECTION_BOOST = 0.2
    RISK_THRESHOLD_HIGH = 0.7
    RISK_THRESHOLD_MEDIUM = 0.4
    FORECAST_DAYS_BACK = 7
    FORECAST_MAX_DETECTIONS = 20
    DEFAULT_TEMPERATURE = 25
    DEFAULT_HUMIDITY = 70
    DEFAULT_RAINFALL = 0
    DEFAULT_WIND_SPEED = 5
    PEST_THRESHOLDS = {
        'rice_bug': {'optimal_temp': (28, 32), 'optimal_humidity': (70, 85)},
        'green_leaf_hopper': {'optimal_temp': (25, 30), 'optimal_humidity': (75, 90)},
        'black_bug': {'optimal_temp': (25, 33), 'optimal_humidity': (80, 95)},
        'brown_plant_hopper': {'optimal_temp': (24, 32), 'optimal_humidity': (75, 90)},
    }
    RAINFALL_MODERATE_MIN = 5
    RAINFALL_MODERATE_MAX = 20
    RAINFALL_HEAVY_THRESHOLD = 20
    WIND_HIGH_THRESHOLD = 15
    TEMP_NEAR_OPTIMAL_RANGE = 5
    HUMIDITY_NEAR_OPTIMAL_RANGE = 10
    FORECAST_CONFIDENCE = 0.8

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Try to import database connection (optional)
try:
    import pymysql
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("pymysql not available - forecasting will use provided weather data only")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_PATHS[0] if MODEL_PATHS else BASE_DIR / 'ml_models' / 'pest_detection' / 'best.pt'
POSSIBLE_MODEL_PATHS = MODEL_PATHS

CLASS_NAMES = [
    'Rice_Bug',
    'black-bug',
    'brown_hopper',
    'green_hopper',
]

# Pest information
PEST_INFO = {
    "Rice_Bug": {
        "scientific_name": "leptocorisa_oratorius",
        "common_name": "Rice Bug (Harabas)",
        "filipino_name": "Harabas",
        "description": "Rice bug that causes grain discoloration and yield loss"
    },
    "black-bug": {
        "scientific_name": "scotinophara_coarctata",
        "common_name": "Rice Black Bug",
        "filipino_name": "Itim na balang",
        "description": "Black bug pest affecting rice during tillering stage"
    },
    "brown_hopper": {
        "scientific_name": "nilaparvata_lugens",
        "common_name": "Brown Planthopper",
        "filipino_name": "Kayumangging balang",
        "description": "Brown plant hopper causing hopperburn"
    },
    "green_hopper": {
        "scientific_name": "nephotettix_virescens",
        "common_name": "Green Leafhopper",
        "filipino_name": "Dahon ng palay",
        "description": "Green leaf hopper that transmits rice tungro virus"
    }
}

# Pesticide recommendations
PESTICIDE_RECS = {
    "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
    "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
    "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
    "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance."
}

# Disease information
PEST_DISEASES = {
    "Rice_Bug": [
        "Grain discoloration and shriveling (sucking damage)",
        "Reduced grain filling leading to yield loss"
    ],
    "black-bug": [
        "Tillering stage damage (sucking), resulting in stunted growth",
        "White earheads in severe infestations"
    ],
    "brown_hopper": [
        "Hopperburn (sudden plant wilting)",
        "Can facilitate sooty mold via honeydew"
    ],
    "green_hopper": [
        "Transmits rice tungro virus (RTV)",
        "Hopperburn under high populations"
    ]
}

# Global model cache
_model_cache = None


# Store the actual model path that was loaded
_loaded_model_path = None

def load_yolo_model():
    """Load YOLO model for pest detection"""
    global _model_cache, _loaded_model_path
    
    if _model_cache is not None:
        return _model_cache
    
    # Try to find model file
    model_path = None
    for path in POSSIBLE_MODEL_PATHS:
        full_path = Path(path).resolve()
        if full_path.exists():
            model_path = full_path
            logger.info(f"📁 Found YOLO model at: {model_path}")
            break
    
    if not model_path:
        error_msg = f"No YOLO model found in Flask folder.\n"
        error_msg += f"Checked locations:\n"
        for i, path in enumerate(POSSIBLE_MODEL_PATHS[:6], 1):  # Show first 6
            error_msg += f"  {i}. {Path(path).resolve()}\n"
        error_msg += f"\nPlace your model file (best.pt) in one of these locations."
        raise FileNotFoundError(error_msg)
    
    _loaded_model_path = str(model_path)
    _model_cache = YOLO(str(model_path))
    logger.info(f"✅ YOLO model loaded successfully from: {model_path}")
    return _model_cache


def aggregate_pest_counts(results, confidence_thresholds: Dict[str, float] = None) -> Dict[str, int]:
    """Aggregate pest counts from YOLO detection results"""
    counts = {name: 0 for name in CLASS_NAMES}
    
    if not results or not results[0] or not hasattr(results[0], 'boxes'):
        return counts
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return counts
    
    # Use thresholds from config (no hardcoded values)
    thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
    
    try:
        for i in range(len(boxes)):
            try:
                cls_idx = int(boxes.cls[i].item()) if hasattr(boxes.cls, 'shape') else int(boxes.cls.tolist()[i])
                conf = float(boxes.conf[i].item()) if hasattr(boxes.conf, 'shape') else float(boxes.conf.tolist()[i])
                
                if 0 <= cls_idx < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_idx]
                    # Apply class-specific confidence filtering (from config)
                    if conf >= thresholds.get(name, CONFIDENCE_FALLBACK):
                        counts[name] += 1
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Error aggregating counts: {e}")
    
    return counts


def extract_detections(results):
    """Extract detection details from YOLO results"""
    detections = []
    
    if not results or not results[0] or not hasattr(results[0], 'boxes'):
        return detections
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections
    
    # Use thresholds from config (no hardcoded values)
    thresholds = CONFIDENCE_THRESHOLDS
    
    try:
        for i in range(len(boxes)):
            try:
                cls_idx = int(boxes.cls[i].item()) if hasattr(boxes.cls, 'shape') else int(boxes.cls.tolist()[i])
                conf = float(boxes.conf[i].item()) if hasattr(boxes.conf, 'shape') else float(boxes.conf.tolist()[i])
                
                if 0 <= cls_idx < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_idx]
                    threshold = thresholds.get(name, CONFIDENCE_FALLBACK)
                    
                    if conf >= threshold:
                        box = boxes.xyxy[i].tolist() if hasattr(boxes, 'xyxy') else [0, 0, 0, 0]
                        detections.append({
                            "class": name,
                            "confidence": round(conf, 3),
                            "bbox": [round(x, 2) for x in box]
                        })
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Error extracting detections: {e}")
    
    return detections


def run_detection(img: Image.Image) -> Dict[str, Any]:
    """Run YOLO detection on image and return results"""
    try:
        model = load_yolo_model()
        
        t0 = time.time()
        # Run inference (all settings from config - no hardcoded values)
        results = model.predict(
            img, 
            imgsz=YOLO_IMAGE_SIZE, 
            conf=YOLO_BASE_CONFIDENCE, 
            iou=YOLO_IOU_THRESHOLD, 
            device=YOLO_DEVICE, 
            verbose=False
        )
        dt = time.time() - t0
        
        # Aggregate counts
        pest_counts = aggregate_pest_counts(results)
        detections = extract_detections(results)
        
        # Get top pest
        total_pests = sum(pest_counts.values())
        if total_pests > 0:
            top_pest = max(pest_counts, key=pest_counts.get)
            top_info = PEST_INFO.get(top_pest, {})
            predicted_class = top_info.get("scientific_name", top_pest)
            common_name = top_info.get("common_name", top_pest)
            filipino_name = top_info.get("filipino_name", "")
            top_conf = detections[0]["confidence"] if detections else 0.0
        else:
            predicted_class = "no_pest_detected"
            common_name = "No Pest Detected"
            filipino_name = ""
            top_conf = 0.0
            top_pest = None
        
        # Filter recommendations to only detected pests
        recommendations = {k: PESTICIDE_RECS.get(k, "") for k, v in pest_counts.items() if v > 0}
        diseases = {k: PEST_DISEASES.get(k, []) for k, v in pest_counts.items() if v > 0}
        
        response = {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": top_conf,
            "common_name": common_name,
            "filipino_name": filipino_name,
            "description": PESTICIDE_RECS.get(top_pest, "") if top_pest else "",
            "treatment": PESTICIDE_RECS.get(top_pest, "") if top_pest else "",
            
            # Detection data
            "total_pests_detected": total_pests,
            "pest_counts": pest_counts,
            "detections": detections,
            "all_predictions": detections,
            "predictions": detections,
            
            # Recommendations
            "diseases": diseases,
            "recommendations": recommendations,
            
            # Metadata
            "inference_time_ms": round(dt * 1000, 1),
            "model": os.path.basename(str(MODEL_PATH)),
            "source": "yolo_detection"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise


@app.route('/health', methods=['GET'])
@app.route('/health/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Try to load model to check if it's available
        model_loaded = False
        model_name = "Not loaded"
        model_path = "Not found"
        try:
            model = load_yolo_model()
            model_loaded = True
            if _loaded_model_path:
                model_path = _loaded_model_path
                model_name = os.path.basename(_loaded_model_path)
            else:
                model_name = "Model loaded (path unknown)"
                model_path = "Unknown"
        except Exception as e:
            model_name = f"Error: {str(e)[:100]}"
            model_path = "Error loading model"
        
        return jsonify({
            "status": "ok" if model_loaded else "warning",
            "model": model_name,
            "model_path": model_path,
            "model_loaded": model_loaded,
            "classes": CLASS_NAMES,
            "api": "Flask",
            "version": "1.0",
            "flask_folder": str(BASE_DIR)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/status', methods=['GET'])
@app.route('/status/', methods=['GET'])
def status_check():
    """Status check (alias for health)"""
    return health_check()


@app.route('/detect', methods=['GET', 'POST'])
@app.route('/detect/', methods=['GET', 'POST'])
def detect():
    """Pest detection endpoint - compatible with Django and Flask APIs"""
    
    # Handle GET request (info)
    if request.method == 'GET':
        return jsonify({
            "endpoint": "Pest Detection API",
            "method": "POST",
            "description": "Upload image file with field name 'image'",
            "example": "curl -X POST -F 'image=@test.jpg' /detect/",
            "classes": CLASS_NAMES
        })
    
    # Handle POST request (detection)
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "error": "No image file provided. Use multipart/form-data with field name 'image'"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "error": "Empty filename"
            }), 400
        
        # Read and process image
        try:
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": f"Invalid image: {e}"
            }), 400
        
        # Run detection
        result = run_detection(img)
        return jsonify(result)
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return jsonify({
            "status": "error",
            "error": f"ML model not found: {e}",
            "message": "Please ensure model file exists at ml_models/pest_detection/best.pt"
        }), 500
    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/classify', methods=['POST'])
@app.route('/classify/', methods=['POST'])
def classify():
    """Classify endpoint for Android app compatibility"""
    return detect()


# ============================================================================
# PEST FORECASTING SYSTEM
# ============================================================================

class SimplePestForecaster:
    """Simple rule-based pest forecasting (no ML dependencies)"""
    
    def __init__(self, db_config=None):
        # Use global DB_CONFIG from config.py or environment variables
        self.db_config = db_config or DB_CONFIG
        
        # Pest types
        self.pest_types = [
            'rice_bug',
            'green_leaf_hopper', 
            'black_bug',
            'brown_plant_hopper'
        ]
        
        # Pest activity thresholds (from config - no hardcoded values)
        self.pest_thresholds = PEST_THRESHOLDS
    
    def get_current_weather(self) -> Dict:
        """Get current weather data from database"""
        if not DB_AVAILABLE:
            return {}
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            query = """
            SELECT 
                temperature,
                humidity,
                wind_speed,
                rainfall_1h,
                pressure,
                cloudiness,
                weather_description,
                location_name,
                last_updated
            FROM weather_current 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            cursor.execute(query)
            weather_data = cursor.fetchone()
            connection.close()
            
            if weather_data:
                return dict(weather_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {}
    
    def get_hourly_weather_forecast(self, days: int = 7) -> List[Dict]:
        """Get hourly weather forecast data for next N days from database or API"""
        if not DB_AVAILABLE:
            return []
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            # Try weather_forecast table first (hourly forecast data from API)
            query = """
            SELECT 
                timestamp,
                temperature,
                humidity,
                wind_speed,
                COALESCE(rainfall_3h, rainfall_1h, 0) as rainfall,
                pressure,
                cloudiness,
                weather_description,
                location_lat,
                location_lon
            FROM weather_forecast 
            WHERE timestamp >= NOW()
            AND timestamp <= DATE_ADD(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (days,))
            forecast_data = cursor.fetchall()
            
            # If no forecast data in database, try to fetch from WeatherAPI.com
            if not forecast_data:
                logger.warning("No forecast data in database. Attempting to fetch from WeatherAPI...")
                forecast_data = self._fetch_forecast_from_api(days)
                
                # If API fetch failed, use current weather as last resort (less accurate)
                if not forecast_data:
                    logger.warning("API fetch failed. Using current weather with estimates (less accurate)")
                    current_query = """
                    SELECT 
                        temperature,
                        humidity,
                        wind_speed,
                        rainfall_1h,
                        pressure,
                        cloudiness,
                        weather_description,
                        location_name
                    FROM weather_current 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                    """
                    cursor.execute(current_query)
                    current = cursor.fetchone()
                    
                    if current:
                        # Generate hourly estimates (NOT ACCURATE - just for fallback)
                        forecast_data = []
                        for hour in range(days * 24):
                            hour_data = dict(current)
                            hour_data['timestamp'] = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:00:00')
                            hour_data['rainfall'] = hour_data.get('rainfall_1h', 0)
                            # Simple temperature variation (not accurate)
                            hour_data['temperature'] = float(hour_data.get('temperature', DEFAULT_TEMPERATURE)) + (hour % 24 - 12) * 0.5
                            forecast_data.append(hour_data)
            
            connection.close()
            
            # Convert to list of dicts
            result = []
            for row in forecast_data:
                result.append(dict(row))
            
            return result
        except Exception as e:
            logger.error(f"Error getting hourly weather forecast: {e}")
            return []
    
    def _fetch_forecast_from_api(self, days: int = 7) -> List[Dict]:
        """Fetch hourly weather forecast from WeatherAPI.com (accurate data)"""
        try:
            import requests
            
                # WeatherAPI.com Configuration (from config.py)
            try:
                from config import WEATHERAPI_KEY, WEATHERAPI_BASE_URL, DEFAULT_LATITUDE, DEFAULT_LONGITUDE
                api_key = WEATHERAPI_KEY
                base_url = WEATHERAPI_BASE_URL
                lat = DEFAULT_LATITUDE
                lon = DEFAULT_LONGITUDE
            except ImportError:
                # Fallback if config not available
                api_key = os.getenv('WEATHERAPI_KEY', '76f91b260dc84341a1733851250710')
                base_url = os.getenv('WEATHERAPI_BASE_URL', 'http://api.weatherapi.com/v1')
                lat = float(os.getenv('DEFAULT_LATITUDE', '10.5379'))
                lon = float(os.getenv('DEFAULT_LONGITUDE', '122.8386'))
            
            url = f"{base_url}/forecast.json"
            params = {
                'key': api_key,
                'q': f"{lat},{lon}",
                'days': min(days, 3),  # WeatherAPI free tier: max 3 days
                'aqi': 'no',
                'alerts': 'no'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_data = []
            
            # Extract hourly forecast data
            for day_data in data['forecast']['forecastday']:
                date = day_data['date']
                for hour in day_data['hour']:
                    hour_time = hour['time'].split()[1] if ' ' in hour['time'] else hour['time']
                    forecast_item = {
                        'timestamp': f"{date} {hour_time}",
                        'temperature': hour['temp_c'],
                        'humidity': hour['humidity'],
                        'wind_speed': hour['wind_kph'] / 3.6,  # Convert to m/s
                        'rainfall': hour.get('precip_mm', 0),
                        'pressure': hour['pressure_mb'],
                        'cloudiness': hour['cloud'],
                        'weather_description': hour['condition']['text'],
                        'location_lat': lat,
                        'location_lon': lon
                    }
                    forecast_data.append(forecast_item)
            
            # If we got less than requested days, extend with last day's pattern
            if len(forecast_data) < days * 24:
                last_hour = forecast_data[-1] if forecast_data else None
                if last_hour:
                    # Extend with similar pattern (less accurate for days 4-7)
                    for day in range(len(forecast_data) // 24, days):
                        for hour in range(24):
                            extended = dict(last_hour)
                            extended['timestamp'] = (datetime.now() + timedelta(days=day, hours=hour)).strftime('%Y-%m-%d %H:00:00')
                            forecast_data.append(extended)
            
            logger.info(f"✅ Fetched {len(forecast_data)} hourly forecast points from WeatherAPI")
            return forecast_data
            
        except ImportError:
            logger.warning("requests library not available. Cannot fetch from API.")
            return []
        except Exception as e:
            logger.error(f"Error fetching forecast from API: {e}")
            return []
    
    def save_forecast_to_database(self, forecast_data: List[Dict]) -> bool:
        """Save hourly forecast data to weather_forecast table"""
        if not DB_AVAILABLE or not forecast_data:
            return False
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Clear old forecast data (older than 1 day)
            cursor.execute("""
                DELETE FROM weather_forecast 
                WHERE timestamp < DATE_SUB(NOW(), INTERVAL 1 DAY)
            """)
            
            # Insert new forecast data
            insert_query = """
                INSERT INTO weather_forecast 
                (timestamp, location_lat, location_lon, temperature, humidity, 
                 pressure, wind_speed, wind_direction, rainfall_3h, cloudiness, 
                 weather_description, weather_main, forecast_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'forecast')
                ON DUPLICATE KEY UPDATE
                    temperature = VALUES(temperature),
                    humidity = VALUES(humidity),
                    pressure = VALUES(pressure),
                    wind_speed = VALUES(wind_speed),
                    rainfall_3h = VALUES(rainfall_3h),
                    cloudiness = VALUES(cloudiness),
                    weather_description = VALUES(weather_description)
            """
            
            saved_count = 0
            for hour_data in forecast_data:
                try:
                    # Parse timestamp
                    timestamp_str = hour_data.get('timestamp', '')
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        timestamp = timestamp_str
                    
                    # Get default coordinates
                    try:
                        default_lat = DEFAULT_LATITUDE
                        default_lon = DEFAULT_LONGITUDE
                    except NameError:
                        default_lat = 10.5379
                        default_lon = 122.8386
                    
                    values = (
                        timestamp,
                        hour_data.get('location_lat', default_lat),
                        hour_data.get('location_lon', default_lon),
                        hour_data.get('temperature'),
                        hour_data.get('humidity'),
                        hour_data.get('pressure'),
                        hour_data.get('wind_speed'),
                        hour_data.get('wind_direction', 0),
                        hour_data.get('rainfall', hour_data.get('rainfall_3h', 0)),
                        hour_data.get('cloudiness', 0),
                        hour_data.get('weather_description', 'Unknown'),
                        'Forecast'
                    )
                    
                    cursor.execute(insert_query, values)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Error saving forecast hour {hour_data.get('timestamp')}: {e}")
                    continue
            
            connection.commit()
            connection.close()
            
            logger.info(f"✅ Saved {saved_count} hourly forecast records to database")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Error saving forecast to database: {e}")
            return False
    
    def update_forecast_automatically(self) -> bool:
        """Automatically fetch and save forecast data from API"""
        try:
            logger.info("🔄 Auto-updating weather forecast...")
            
            # Fetch forecast from API (3 days max for free tier)
            forecast_data = self._fetch_forecast_from_api(days=3)
            
            if forecast_data:
                # Save to database
                if self.save_forecast_to_database(forecast_data):
                    logger.info("✅ Forecast automatically updated successfully")
                    return True
                else:
                    logger.error("❌ Failed to save forecast to database")
                    return False
            else:
                logger.warning("⚠️ No forecast data fetched from API")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in automatic forecast update: {e}")
            return False
    
    def get_recent_pest_data(self, days: int = None) -> List[Dict]:
        """Get recent pest detection data"""
        # Use config value if not provided
        if days is None:
            days = FORECAST_DAYS_BACK
        if not DB_AVAILABLE:
            return []
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            query = """
            SELECT 
                created_at,
                classification_json
            FROM images_inbox 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND classification_json IS NOT NULL 
            AND classification_json != ''
            ORDER BY created_at DESC
            LIMIT %s
            """
            
            cursor.execute(query, (days, FORECAST_MAX_DETECTIONS))
            results = cursor.fetchall()
            connection.close()
            
            pest_data = []
            for row in results:
                try:
                    classification = json.loads(row['classification_json'])
                    if 'pest_counts' in classification:
                        pest_data.append({
                            'date': str(row['created_at']),
                            'pest_counts': classification['pest_counts']
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return pest_data
        except Exception as e:
            logger.error(f"Error getting pest data: {e}")
            return []
    
    def calculate_pest_risk(self, pest_type: str, weather_data: Dict) -> Dict:
        """Calculate pest risk based on weather conditions"""
        if pest_type not in self.pest_thresholds:
            return {
                'risk_level': 'unknown', 
                'risk_score': 0.5, 
                'confidence': FORECAST_CONFIDENCE  # From config
            }
        
        thresholds = self.pest_thresholds[pest_type]
        # Use default values from config (no hardcoded values)
        temp = weather_data.get('temperature', DEFAULT_TEMPERATURE)
        humidity = weather_data.get('humidity', DEFAULT_HUMIDITY)
        rainfall = weather_data.get('rainfall_1h', DEFAULT_RAINFALL)
        wind_speed = weather_data.get('wind_speed', DEFAULT_WIND_SPEED)
        
        risk_score = RISK_BASE_SCORE  # From config
        
        # Temperature factor (from config)
        temp_min, temp_max = thresholds['optimal_temp']
        if temp_min <= temp <= temp_max:
            risk_score += RISK_TEMP_OPTIMAL
        elif abs(temp - (temp_min + temp_max) / 2) <= TEMP_NEAR_OPTIMAL_RANGE:
            risk_score += RISK_TEMP_NEAR
        
        # Humidity factor (from config)
        humidity_min, humidity_max = thresholds['optimal_humidity']
        if humidity_min <= humidity <= humidity_max:
            risk_score += RISK_HUMIDITY_OPTIMAL
        elif humidity >= humidity_min - HUMIDITY_NEAR_OPTIMAL_RANGE:
            risk_score += RISK_HUMIDITY_NEAR
        
        # Rainfall factor (from config)
        if RAINFALL_MODERATE_MIN < rainfall <= RAINFALL_MODERATE_MAX:
            risk_score += RISK_RAINFALL_MODERATE
        elif rainfall > RAINFALL_HEAVY_THRESHOLD:
            risk_score += RISK_RAINFALL_HEAVY
        
        # Wind factor (from config)
        if wind_speed > WIND_HIGH_THRESHOLD:
            risk_score += RISK_WIND_HIGH
        
        # Ensure risk score is between 0 and 1
        risk_score = max(0.1, min(1.0, risk_score))
        
        # Determine risk level (from config)
        if risk_score >= RISK_THRESHOLD_HIGH:
            risk_level = 'high'
        elif risk_score >= RISK_THRESHOLD_MEDIUM:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': FORECAST_CONFIDENCE  # From config
        }
    
    def generate_forecast(self, weather_data: Dict = None, days: int = 7) -> Dict:
        """Generate comprehensive pest forecast for 7 days with hourly data"""
        # Get hourly weather forecast for 7 days
        hourly_weather = self.get_hourly_weather_forecast(days)
        
        if not hourly_weather:
            # Fallback to current weather if no forecast available
            if not weather_data:
                weather_data = self.get_current_weather()
            
            if not weather_data:
                return {'error': 'No weather data available'}
            
            # Generate single forecast from current weather
            return self._generate_single_forecast(weather_data)
        
        # Get recent pest data
        recent_pests = self.get_recent_pest_data(7)
        
        # Group hourly data by day
        daily_forecasts = {}
        for hour_data in hourly_weather:
            # Extract date from timestamp
            timestamp_str = hour_data.get('timestamp', '')
            try:
                if isinstance(timestamp_str, str):
                    hour_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                else:
                    hour_dt = timestamp_str
                date_key = hour_dt.strftime('%Y-%m-%d')
            except:
                date_key = datetime.now().strftime('%Y-%m-%d')
            
            if date_key not in daily_forecasts:
                daily_forecasts[date_key] = {
                    'date': date_key,
                    'hours': []
                }
            
            # Get hour from parsed datetime or default to 0
            try:
                hour_value = hour_dt.hour if 'hour_dt' in locals() and hour_dt else 0
            except:
                hour_value = 0
            
            daily_forecasts[date_key]['hours'].append({
                'hour': hour_value,
                'timestamp': timestamp_str,
                'temperature': hour_data.get('temperature'),
                'humidity': hour_data.get('humidity'),
                'rainfall': hour_data.get('rainfall', hour_data.get('rainfall_1h', 0)),
                'wind_speed': hour_data.get('wind_speed', 0),
                'weather_description': hour_data.get('weather_description', 'Unknown')
            })
        
        # Generate risk assessment for each day
        forecast_days = []
        for date_key in sorted(daily_forecasts.keys())[:days]:
            day_data = daily_forecasts[date_key]
            
            # Calculate daily averages from hourly data
            hours = day_data['hours']
            if not hours:
                continue
            
            avg_temp = sum(h.get('temperature', DEFAULT_TEMPERATURE) for h in hours) / len(hours)
            avg_humidity = sum(h.get('humidity', DEFAULT_HUMIDITY) for h in hours) / len(hours)
            total_rainfall = sum(h.get('rainfall', 0) for h in hours)
            avg_wind = sum(h.get('wind_speed', 0) for h in hours) / len(hours)
            
            # Use average weather for daily risk calculation
            daily_weather = {
                'temperature': avg_temp,
                'humidity': avg_humidity,
                'rainfall_1h': total_rainfall / len(hours),  # Average hourly rainfall
                'wind_speed': avg_wind,
                'weather_description': hours[0].get('weather_description', 'Unknown')
            }
            
            # Calculate pest risks for this day
            pest_risks = {}
            for pest_type in self.pest_types:
                risk = self.calculate_pest_risk(pest_type, daily_weather)
                
                # Adjust based on recent detections
                if recent_pests:
                    recent_count = 0
                    for detection in recent_pests:
                        pest_counts = detection.get('pest_counts', {})
                        # Map pest type names
                        pest_key = pest_type.replace('_', '-')
                        if pest_key == 'rice-bug':
                            pest_key = 'Rice_Bug'
                        elif pest_key == 'black-bug':
                            pest_key = 'black-bug'
                        elif pest_key == 'brown-plant-hopper':
                            pest_key = 'brown_hopper'
                        elif pest_key == 'green-leaf-hopper':
                            pest_key = 'green_hopper'
                        
                        recent_count += pest_counts.get(pest_key, 0)
                    
                    if recent_count > 0:
                        risk['risk_score'] = min(1.0, risk['risk_score'] + RISK_RECENT_DETECTION_BOOST)
                        if risk['risk_score'] >= RISK_THRESHOLD_HIGH:
                            risk['risk_level'] = 'high'
                        elif risk['risk_score'] >= RISK_THRESHOLD_MEDIUM:
                            risk['risk_level'] = 'medium'
                
                pest_risks[pest_type] = risk
            
            # Calculate overall risk for the day
            risk_scores = [risk['risk_score'] for risk in pest_risks.values()]
            overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
            
            if overall_risk_score >= RISK_THRESHOLD_HIGH:
                overall_level = 'high'
            elif overall_risk_score >= RISK_THRESHOLD_MEDIUM:
                overall_level = 'medium'
            else:
                overall_level = 'low'
            
            forecast_days.append({
                'date': date_key,
                'overall_risk': {
                    'level': overall_level,
                    'score': round(overall_risk_score, 2)
                },
                'pest_risks': pest_risks,
                'weather': {
                    'temperature_avg': round(avg_temp, 1),
                    'humidity_avg': round(avg_humidity, 1),
                    'rainfall_total': round(total_rainfall, 2),
                    'wind_speed_avg': round(avg_wind, 1),
                    'weather_description': daily_weather['weather_description']
                },
                'hourly_data': hours
            })
        
        # Calculate overall 7-day risk summary
        all_risk_scores = []
        for day in forecast_days:
            all_risk_scores.append(day['overall_risk']['score'])
        
        overall_7day_risk = sum(all_risk_scores) / len(all_risk_scores) if all_risk_scores else 0.5
        
        if overall_7day_risk >= RISK_THRESHOLD_HIGH:
            overall_7day_level = 'high'
        elif overall_7day_risk >= RISK_THRESHOLD_MEDIUM:
            overall_7day_level = 'medium'
        else:
            overall_7day_level = 'low'
        
        return {
            'generated_at': datetime.now().isoformat(),
            'location': hourly_weather[0].get('location_name', DEFAULT_LOCATION) if hourly_weather else DEFAULT_LOCATION,
            'forecast_period': f'{days} days',
            'overall_7day_risk': {
                'level': overall_7day_level,
                'score': round(overall_7day_risk, 2)
            },
            'daily_forecasts': forecast_days,
            'recent_detections': len(recent_pests)
        }
    
    def _generate_single_forecast(self, weather_data: Dict) -> Dict:
        """Generate single forecast from current weather (fallback method)"""
        recent_pests = self.get_recent_pest_data(7)
        
        # Calculate pest risks
        pest_risks = {}
        for pest_type in self.pest_types:
            risk = self.calculate_pest_risk(pest_type, weather_data)
            
            # Adjust based on recent detections
            if recent_pests:
                recent_count = 0
                for detection in recent_pests:
                    pest_counts = detection.get('pest_counts', {})
                    pest_key = pest_type.replace('_', '-')
                    if pest_key == 'rice-bug':
                        pest_key = 'Rice_Bug'
                    elif pest_key == 'black-bug':
                        pest_key = 'black-bug'
                    elif pest_key == 'brown-plant-hopper':
                        pest_key = 'brown_hopper'
                    elif pest_key == 'green-leaf-hopper':
                        pest_key = 'green_hopper'
                    
                    recent_count += pest_counts.get(pest_key, 0)
                
                if recent_count > 0:
                    risk['risk_score'] = min(1.0, risk['risk_score'] + RISK_RECENT_DETECTION_BOOST)
                    if risk['risk_score'] >= RISK_THRESHOLD_HIGH:
                        risk['risk_level'] = 'high'
                    elif risk['risk_score'] >= RISK_THRESHOLD_MEDIUM:
                        risk['risk_level'] = 'medium'
            
            pest_risks[pest_type] = risk
        
        # Calculate overall risk
        risk_scores = [risk['risk_score'] for risk in pest_risks.values()]
        overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        
        if overall_risk_score >= RISK_THRESHOLD_HIGH:
            overall_level = 'high'
        elif overall_risk_score >= RISK_THRESHOLD_MEDIUM:
            overall_level = 'medium'
        else:
            overall_level = 'low'
        
        return {
            'generated_at': datetime.now().isoformat(),
            'location': weather_data.get('location_name', DEFAULT_LOCATION),
            'current_weather': {
                'temperature': weather_data.get('temperature'),
                'humidity': weather_data.get('humidity'),
                'rainfall': weather_data.get('rainfall_1h', 0),
                'wind_speed': weather_data.get('wind_speed', 0),
                'weather_description': weather_data.get('weather_description', 'Unknown')
            },
            'overall_risk': {
                'level': overall_level,
                'score': round(overall_risk_score, 2)
            },
            'pest_risks': pest_risks,
            'recent_detections': len(recent_pests)
        }


# Initialize forecaster
_forecaster = None

def get_forecaster():
    """Get or create forecaster instance"""
    global _forecaster
    if _forecaster is None:
        _forecaster = SimplePestForecaster()
    return _forecaster


@app.route('/forecast', methods=['GET', 'POST'])
@app.route('/forecast/', methods=['GET', 'POST'])
def forecast():
    """Generate 7-day pest forecast with hourly weather data"""
    try:
        forecaster = get_forecaster()
        
        # Get number of days from request (default 7)
        days = 7
        if request.method == 'POST':
            data = request.get_json() or {}
            days = int(data.get('days', 7))
        elif request.method == 'GET':
            days = int(request.args.get('days', 7))
        
        # Limit to reasonable range
        days = max(1, min(days, 14))  # Between 1 and 14 days
        
        # Generate 7-day forecast with hourly data
        forecast_result = forecaster.generate_forecast(days=days)
        
        if 'error' in forecast_result:
            return jsonify(forecast_result), 400
        
        return jsonify({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/forecast/quick', methods=['GET', 'POST'])
@app.route('/forecast/quick/', methods=['GET', 'POST'])
def quick_forecast():
    """Quick forecast with minimal data"""
    try:
        forecaster = get_forecaster()
        
        # Get weather from request or use defaults
        if request.method == 'POST':
            data = request.get_json() or {}
            weather_data = data.get('weather', {})
        else:
            # Try to get from database
            weather_data = forecaster.get_current_weather()
            if not weather_data:
                # Use defaults
                # Use default values from config (no hardcoded values)
                weather_data = {
                    'temperature': DEFAULT_TEMPERATURE,
                    'humidity': DEFAULT_HUMIDITY,
                    'rainfall_1h': DEFAULT_RAINFALL,
                    'wind_speed': DEFAULT_WIND_SPEED,
                    'location_name': DEFAULT_LOCATION
                }
        
        # Generate forecast (use single forecast method for quick forecast)
        forecast_result = forecaster._generate_single_forecast(weather_data)
        
        if 'error' in forecast_result:
            return jsonify(forecast_result), 400
        
        return jsonify({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Quick forecast error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/forecast/current', methods=['GET'])
@app.route('/forecast/current/', methods=['GET'])
def current_forecast():
    """Get current forecast from database or generate new (7-day forecast)"""
    try:
        forecaster = get_forecaster()
        # Get days parameter (default 7)
        days = int(request.args.get('days', 7))
        days = max(1, min(days, 14))
        
        forecast_result = forecaster.generate_forecast(days=days)
        
        if 'error' in forecast_result:
            return jsonify(forecast_result), 400
        
        return jsonify({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Current forecast error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/forecast/update', methods=['POST', 'GET'])
@app.route('/forecast/update/', methods=['POST', 'GET'])
def update_forecast():
    """Manually trigger forecast update from API"""
    try:
        forecaster = get_forecaster()
        success = forecaster.update_forecast_automatically()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Forecast updated successfully from WeatherAPI'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update forecast'
            }), 500
    except Exception as e:
        logger.error(f"Forecast update error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """API information"""
    return jsonify({
        "api": "AgriShield ML Flask API",
        "version": "1.0",
        "modules": {
            "pest_detection": "YOLO-based pest detection",
            "pest_forecasting": "Rule-based pest forecasting"
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /status": "Status check",
            "POST /detect": "Detect pests in image (multipart/form-data: image=file)",
            "POST /classify": "Classify pests (alias for /detect)",
            "GET /forecast": "Generate 7-day pest forecast with hourly weather data",
            "POST /forecast": "Generate 7-day pest forecast (optional: days parameter)",
            "GET /forecast/quick": "Quick forecast",
            "GET /forecast/current": "Get current forecast",
            "POST /forecast/update": "Manually update forecast from WeatherAPI"
        },
        "classes": CLASS_NAMES
    })


def auto_update_forecast_worker():
    """Background worker to automatically update forecast every 6 hours"""
    # Wait 30 seconds after Flask starts before first update
    time.sleep(30)
    
    while True:
        try:
            forecaster = get_forecaster()
            forecaster.update_forecast_automatically()
        except Exception as e:
            logger.error(f"Error in auto-update worker: {e}")
        
        # Update every 6 hours (21600 seconds)
        time.sleep(21600)


if __name__ == '__main__':
    print("=" * 70)
    print("🌾 AgriShield ML Flask API")
    print("=" * 70)
    print(f"Model path: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print()
    print(f"Starting Flask API on port {FLASK_PORT}...")
    print()
    print("📡 Endpoints:")
    print("  Detection:")
    print("    GET  /health        - Health check")
    print("    GET  /status        - Status check")
    print("    POST /detect        - Detect pests (multipart form-data)")
    print("    POST /classify     - Classify pests (Android app)")
    print()
    print("  Forecasting:")
    print("    GET  /forecast      - Generate 7-day pest forecast (hourly weather data)")
    print("    POST /forecast      - Generate 7-day forecast (optional: days parameter)")
    print("    GET  /forecast/quick - Quick single forecast")
    print("    GET  /forecast/current - Get 7-day forecast from database")
    print("    POST /forecast/update - Manually update forecast from API")
    print()
    print("🔄 Auto-Update: Forecast updates automatically every 6 hours")
    print()
    print(f"Access from LAN: http://your-server-ip:{FLASK_PORT}")
    print(f"Local access: http://localhost:{FLASK_PORT}")
    print("=" * 70)
    
    # Start background thread for automatic forecast updates
    try:
        update_thread = threading.Thread(target=auto_update_forecast_worker, daemon=True)
        update_thread.start()
        logger.info("✅ Auto-update forecast worker started (updates every 6 hours)")
    except Exception as e:
        logger.warning(f"⚠️ Could not start auto-update worker: {e}")
    
    # Run Flask app
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)


