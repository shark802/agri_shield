"""
Django Views - Converted from Flask app.py
ML Pest Detection and Forecasting API
"""
import os
import json
import logging
from pathlib import Path
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image
import io

from .ml_utils import (
    get_model, get_class_names, load_yolo_model,
    YOLO_IMAGE_SIZE, YOLO_BASE_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_DEVICE,
    CONFIDENCE_THRESHOLDS, CONFIDENCE_FALLBACK, _loaded_model_path
)

logger = logging.getLogger(__name__)

# Try to import database connection
try:
    import pymysql
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("pymysql not available - forecasting will use provided weather data only")


@require_http_methods(["GET"])
def index(request):
    """API information"""
    class_names = get_class_names()
    return JsonResponse({
        "api": "AgriShield ML Django API",
        "version": "1.0",
        "modules": {
            "pest_detection": "YOLO-based pest detection",
            "pest_forecasting": "Rule-based pest forecasting"
        },
        "endpoints": {
            "GET /health/": "Health check",
            "GET /status/": "Status check",
            "POST /detect/": "Detect pests in image (multipart/form-data: image=file)",
            "POST /classify/": "Classify pests (alias for /detect/)",
            "GET /forecast/": "Generate 7-day pest forecast with hourly weather data",
            "POST /forecast/": "Generate 7-day pest forecast (optional: days parameter)",
            "GET /forecast/quick/": "Quick forecast",
            "GET /forecast/current/": "Get current forecast",
            "POST /forecast/update/": "Manually update forecast from WeatherAPI"
        },
        "classes": class_names
    })


@require_http_methods(["GET"])
def health(request):
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
        
        class_names = get_class_names()
        return JsonResponse({
            "status": "ok" if model_loaded else "warning",
            "model": model_name,
            "model_path": model_path,
            "model_loaded": model_loaded,
            "classes": class_names if class_names else ["Loading..."],
            "num_classes": len(class_names) if class_names else 0,
            "api": "Django",
            "version": "1.0",
        })
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def status(request):
    """Status check endpoint"""
    class_names = get_class_names()
    return JsonResponse({
        "status": "running",
        "classes": class_names,
        "num_classes": len(class_names),
        "api": "Django"
    })


@csrf_exempt
@require_http_methods(["GET", "POST"])
def detect(request):
    """
    Detect pests in uploaded image
    POST: multipart/form-data with 'image' field
    GET: Returns endpoint information
    """
    if request.method == 'GET':
        return JsonResponse({
            "endpoint": "/detect/",
            "method": "POST",
            "description": "Detect pests in image",
            "parameters": {
                "image": "Image file (multipart/form-data)"
            },
            "returns": "JSON with pest detections, counts, and bounding boxes"
        })
    
    try:
        # Get image from request
        if 'image' not in request.FILES:
            return JsonResponse({
                "error": "No image file provided. Use 'image' field in multipart/form-data."
            }, status=400)
        
        image_file = request.FILES['image']
        
        # Read image
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Load model
        model = get_model()
        
        # Run inference
        results = model.predict(
            image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_BASE_CONFIDENCE,
            iou=YOLO_IOU_THRESHOLD,
            device=YOLO_DEVICE,
            verbose=False
        )
        
        # Process results
        detections = []
        counts = {}
        class_names = get_class_names()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                
                # Apply confidence threshold
                threshold = CONFIDENCE_THRESHOLDS.get(class_name, CONFIDENCE_FALLBACK)
                if conf >= threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })
                    counts[class_name] = counts.get(class_name, 0) + 1
        
        return JsonResponse({
            "success": True,
            "detections": detections,
            "counts": counts,
            "total_detections": len(detections),
            "classes_detected": list(counts.keys())
        })
        
    except Exception as e:
        logger.error(f"Error in detect: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def classify(request):
    """Classify pests (alias for /detect/)"""
    return detect(request)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def forecast(request):
    """Generate 7-day pest forecast with hourly weather data"""
    try:
        from .forecasting import get_forecaster
        
        forecaster = get_forecaster()
        
        # Get number of days from request (default 7)
        days = 7
        if request.method == 'POST':
            try:
                import json
                data = json.loads(request.body) if request.body else {}
                days = int(data.get('days', 7))
            except:
                days = 7
        elif request.method == 'GET':
            days = int(request.GET.get('days', 7))
        
        # Limit to reasonable range
        days = max(1, min(days, 14))  # Between 1 and 14 days
        
        # Generate forecast
        forecast_result = forecaster.generate_forecast(days=days)
        
        if 'error' in forecast_result:
            return JsonResponse(forecast_result, status=400)
        
        return JsonResponse({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def forecast_quick(request):
    """Quick forecast with minimal data"""
    try:
        from .forecasting import get_forecaster, DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY, DEFAULT_RAINFALL, DEFAULT_WIND_SPEED, DEFAULT_LOCATION
        
        forecaster = get_forecaster()
        
        # Get weather from request or use defaults
        if request.method == 'POST':
            try:
                import json
                data = json.loads(request.body) if request.body else {}
                weather_data = data.get('weather', {})
            except:
                weather_data = {}
        else:
            weather_data = forecaster.get_current_weather()
            if not weather_data:
                weather_data = {
                    'temperature': DEFAULT_TEMPERATURE,
                    'humidity': DEFAULT_HUMIDITY,
                    'rainfall_1h': DEFAULT_RAINFALL,
                    'wind_speed': DEFAULT_WIND_SPEED,
                    'location_name': DEFAULT_LOCATION
                }
        
        # Generate forecast
        forecast_result = forecaster._generate_single_forecast(weather_data)
        
        if 'error' in forecast_result:
            return JsonResponse(forecast_result, status=400)
        
        return JsonResponse({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Quick forecast error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def forecast_current(request):
    """Get current forecast from database or generate new (7-day forecast)"""
    try:
        from .forecasting import get_forecaster
        
        forecaster = get_forecaster()
        days = int(request.GET.get('days', 7))
        days = max(1, min(days, 14))
        
        forecast_result = forecaster.generate_forecast(days=days)
        
        if 'error' in forecast_result:
            return JsonResponse(forecast_result, status=400)
        
        return JsonResponse({
            'status': 'success',
            **forecast_result
        })
    except Exception as e:
        logger.error(f"Current forecast error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST", "GET"])
def forecast_update(request):
    """Manually trigger forecast update from API"""
    try:
        from .forecasting import get_forecaster
        
        forecaster = get_forecaster()
        success = forecaster.update_forecast_automatically()
        
        if success:
            return JsonResponse({
                'status': 'success',
                'message': 'Forecast updated successfully from WeatherAPI'
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to update forecast'
            }, status=500)
    except Exception as e:
        logger.error(f"Forecast update error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def test_db(request):
    """Test database connection and images_inbox table"""
    try:
        if not DB_AVAILABLE:
            return JsonResponse({
                "status": "error",
                "message": "pymysql not available"
            }, status=500)
        
        try:
            from config import DB_CONFIG
            import pymysql
            
            connection = pymysql.connect(**DB_CONFIG)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            # Test basic connection
            cursor.execute("SELECT 1 as test")
            test_result = cursor.fetchone()
            
            # Test images_inbox table
            cursor.execute("""
                SELECT COUNT(*) as total_images,
                       COUNT(CASE WHEN classification_json IS NOT NULL AND classification_json != '' THEN 1 END) as classified_images
                FROM images_inbox
            """)
            table_info = cursor.fetchone()
            
            # Get recent images
            cursor.execute("""
                SELECT id, created_at, classification_json IS NOT NULL as has_classification
                FROM images_inbox
                ORDER BY created_at DESC
                LIMIT 5
            """)
            recent_images = cursor.fetchall()
            
            connection.close()
            
            return JsonResponse({
                "status": "success",
                "database": "connected",
                "test_query": test_result,
                "images_inbox": {
                    "total_images": table_info['total_images'],
                    "classified_images": table_info['classified_images']
                },
                "recent_images": list(recent_images)
            })
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "error": str(e)
        }, status=500)

