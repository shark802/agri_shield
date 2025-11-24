"""
ML Model Loading Utilities for Django
Copied from Flask app.py to maintain same functionality
"""
import os
import logging
from pathlib import Path
from ultralytics import YOLO
import requests

logger = logging.getLogger(__name__)

# Try to load configuration
try:
    from config import (
        MODEL_PATHS, PHP_BASE_URL,
        YOLO_IMAGE_SIZE, YOLO_BASE_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_DEVICE,
        CONFIDENCE_THRESHOLDS, CONFIDENCE_FALLBACK
    )
    USE_CONFIG_FILE = True
except ImportError:
    USE_CONFIG_FILE = False
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_PATHS = [
        BASE_DIR / 'datasets' / 'best 2.pt',
        BASE_DIR / 'pest_detection_ml' / 'models' / 'best.pt',
        BASE_DIR / 'models' / 'best.pt',
    ]
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
    PHP_BASE_URL = os.getenv('PHP_BASE_URL', 'http://localhost/Proto1')

# Global model and class names
_loaded_model = None
_loaded_model_path = None
CLASS_NAMES = []


def get_active_model_path() -> str:
    """Fetch the currently active model path from the database"""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    try:
        # Get PHP base URL from config
        php_base = PHP_BASE_URL
        
        # Call PHP endpoint to get active model
        php_url = f"{php_base}/pest_detection_ml/api/get_active_model_path.php"
        response = requests.get(php_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                model_path = data.get('model_path')
                if model_path and os.path.exists(model_path):
                    logger.info(f"✅ Using active model: {os.path.basename(model_path)}")
                    return model_path
    except Exception as e:
        logger.warning(f"⚠️ Could not get active model from database: {e}")
    
    # Fallback to default model - using "best 2.pt" from datasets folder
    datasets_model = BASE_DIR / "datasets" / "best 2.pt"
    if datasets_model.exists():
        logger.info(f"✅ Using default model: best 2.pt (from datasets folder)")
        return str(datasets_model)
    
    # Secondary fallback to models folder
    fallback = BASE_DIR / "pest_detection_ml" / "models" / "best.pt"
    if fallback.exists():
        logger.info(f"⚠️ Using fallback model: best.pt (from models folder)")
        return str(fallback)
    
    # Try MODEL_PATHS as last resort
    for path in MODEL_PATHS:
        full_path = Path(path).resolve()
        if full_path.exists():
            logger.info(f"✅ Using model from MODEL_PATHS: {full_path}")
            return str(full_path)
    
    # If neither exists, return datasets path (will show error when loading)
    logger.error(f"❌ Model not found: {datasets_model}")
    return str(datasets_model)


def load_yolo_model():
    """Load YOLO model (lazy loading - only loads when needed)"""
    global _loaded_model, _loaded_model_path, CLASS_NAMES
    
    # If already loaded, return it
    if _loaded_model is not None:
        return _loaded_model
    
    # Get model path
    model_path = get_active_model_path()
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"🔄 Loading YOLO model: {model_path}")
        _loaded_model = YOLO(model_path)
        _loaded_model_path = model_path
        
        # Get class names from model
        if hasattr(_loaded_model, 'names'):
            CLASS_NAMES = list(_loaded_model.names.values())
        else:
            CLASS_NAMES = []
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
        
        return _loaded_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def get_model():
    """Get the loaded model (loads if not already loaded)"""
    return load_yolo_model()


def get_class_names():
    """Get class names from loaded model"""
    if not CLASS_NAMES:
        load_yolo_model()
    return CLASS_NAMES

