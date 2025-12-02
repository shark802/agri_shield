#!/usr/bin/env python3
"""
Combined AgriShield API - Pest Detection + Training
ONNX Runtime for inference + PyTorch for training
"""

from __future__ import annotations

import io
import os
import time
import numpy as np
import json
import subprocess
import threading
from typing import Dict, Any
from pathlib import Path

# Set environment variables to disable GUI dependencies (for Heroku)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available. Install with: pip install onnxruntime")

# Database for training - using PHP API instead of direct MySQL
import requests

app = Flask(__name__)
CORS(app)

# ============================================================================
# PEST DETECTION (ONNX Runtime)
# ============================================================================

# Model configuration
ONNX_MODEL_PATH = None
CLASS_NAMES = []
session = None
input_details = None
output_details = None

# Detection thresholds (configurable via environment variables)
DETECTION_CONF_THRESHOLD = float(os.getenv('DETECTION_CONF_THRESHOLD', '0.25'))  # Base confidence threshold (25%)
CLASSIFICATION_MIN_THRESHOLD = float(os.getenv('CLASSIFICATION_MIN_THRESHOLD', '0.5'))  # Minimum for classification (50%)
CONFIDENCE_GAP_REQUIREMENT = float(os.getenv('CONFIDENCE_GAP_REQUIREMENT', '0.2'))  # Required gap between top classes (20%)
YOLO_CONF_THRESHOLD = float(os.getenv('YOLO_CONF_THRESHOLD', '0.35'))  # Minimum for YOLO (35%)

print(f"📊 Detection thresholds configured:")
print(f"   Base threshold: {DETECTION_CONF_THRESHOLD}")
print(f"   Classification min: {CLASSIFICATION_MIN_THRESHOLD}")
print(f"   Confidence gap: {CONFIDENCE_GAP_REQUIREMENT}")
print(f"   YOLO threshold: {YOLO_CONF_THRESHOLD}")

def find_onnx_model() -> str:
    """Find ONNX model file - checks local files and downloads from server if needed"""
    base_dir = Path(__file__).resolve().parent
    
    # Priority order - check models/ directory first (for Heroku deployment)
    # "best 2.onnx" is the default model from git repository
    candidates = [
        base_dir / "models" / "best 2.onnx",  # Default model from git (highest priority)
        base_dir / "models" / "best.onnx",    # Trained models copy here
        base_dir / "models" / "best5.onnx",
        base_dir / "deployment" / "models" / "best 2.onnx",
        base_dir / "deployment" / "models" / "best.onnx",
        base_dir / "deployment" / "models" / "best5.onnx",
        base_dir / "datasets" / "best 2.onnx",
        base_dir / "datasets" / "best.onnx",
        base_dir / "datasets" / "best5.onnx",
        base_dir / "pest_detection_ml" / "models" / "best.onnx",
    ]
    
    # Check standard locations first
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # If no standard model found, check job directories (for trained models on Heroku)
    models_dir = base_dir / "models"
    if models_dir.exists():
        # Find all job_* directories
        job_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('job_')], 
                         reverse=True)  # Most recent first
        
        for job_dir in job_dirs:
            # Look for best_model*.onnx files in job directory
            onnx_files = sorted(job_dir.glob('best_model*.onnx'), 
                              key=lambda p: p.stat().st_mtime, 
                              reverse=True)  # Most recent first
            if onnx_files:
                print(f"📦 Found trained model in {job_dir.name}: {onnx_files[0].name}")
                return str(onnx_files[0])
    
    # If still not found and we're on Heroku, try to get active model from web server
    # (Models are uploaded to web server during training, but Heroku needs local copy)
    php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')
    try:
        import requests
        # Try to get active model info from database via PHP API
        # For now, just raise error - model should be uploaded to Heroku during training
        pass
    except:
        pass
    
    raise FileNotFoundError(
        f"ONNX model not found. Checked: {[str(c) for c in candidates]} and job directories. "
        f"Note: On Heroku, models must be uploaded during training or placed in models/ directory."
    )

def load_onnx_model(model_path: str):
    """Load ONNX model"""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")
    
    print(f"📦 Loading ONNX model: {Path(model_path).name}")
    
    # Create inference session
    sess = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']  # Use CPU
    )
    
    # Get input/output details
    input_details = sess.get_inputs()[0]
    output_details = sess.get_outputs()[0]
    
    print(f"✅ Model loaded")
    print(f"   Input: {input_details.name}, Shape: {input_details.shape}")
    print(f"   Output: {output_details.name}, Shape: {output_details.shape}")
    
    return sess, input_details, output_details

# Initialize model
if ONNX_AVAILABLE:
    try:
        ONNX_MODEL_PATH = find_onnx_model()
        print(f"🔍 Found model at: {ONNX_MODEL_PATH}")
        session, input_details, output_details = load_onnx_model(ONNX_MODEL_PATH)
        
        # Default class names (update based on your model)
        CLASS_NAMES = [
            "Rice_Bug",
            "White stem borer",
            "black-bug",
            "brown_hopper",
            "green_hopper",
        ]
        print(f"✅ Model loaded successfully: {Path(ONNX_MODEL_PATH).name}")
        print(f"   Classes: {CLASS_NAMES}")
    except Exception as e:
        import traceback
        print(f"⚠️  Could not load ONNX model: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        session = None
        input_details = None
        output_details = None
else:
    session = None
    input_details = None
    output_details = None

# ============================================================================
# TRAINING SERVICE - Using PHP API Gateway (No Direct Database Access)
# ============================================================================

# PHP API Base URL - Heroku calls PHP endpoints instead of MySQL directly
PHP_API_BASE = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')

# Training script path
TRAINING_SCRIPT = os.getenv('TRAINING_SCRIPT', 'train.py')

def get_training_job(job_id):
    """Get training job via PHP API"""
    try:
        url = f"{PHP_API_BASE}/get_job.php"
        response = requests.get(url, params={'job_id': job_id}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('job'):
                return data['job']
        elif response.status_code == 404:
            print(f"Job {job_id} not found")
        else:
            print(f"Error getting job: HTTP {response.status_code} - {response.text}")
        return None
    except Exception as e:
        print(f"Error getting job: {e}")
        return None

def update_job_status(job_id, status, message=None):
    """Update training job status via PHP API"""
    try:
        url = f"{PHP_API_BASE}/update_status.php"
        data = {
            'job_id': job_id,
            'status': status
        }
        if message:
            data['message'] = message[:500]
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if not result.get('success'):
                print(f"Failed to update status: {result.get('error')}")
        else:
            print(f"Error updating status: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error updating job status: {e}")

def log_to_database(job_id, level, message):
    """Log to database via PHP API"""
    try:
        url = f"{PHP_API_BASE}/add_log.php"
        data = {
            'job_id': job_id,
            'level': level,
            'message': message[:1000]
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code != 200:
            print(f"Error logging: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Log error: {e}")

def run_training(job_id):
    """Run training in background thread"""
    try:
        job = get_training_job(job_id)
        if not job:
            print(f"Job {job_id} not found")
            return
        
        # training_config might be a string (JSON) or already a dict
        training_config_raw = job.get('training_config', '{}')
        if isinstance(training_config_raw, str):
            config = json.loads(training_config_raw)
        elif isinstance(training_config_raw, dict):
            config = training_config_raw
        else:
            config = {}
        
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 8)
        
        update_job_status(job_id, 'running')
        log_to_database(job_id, 'INFO', 'Training started on Heroku service')
        
        # Check if training script exists
        script_path = os.path.join(os.path.dirname(__file__), TRAINING_SCRIPT)
        if not os.path.exists(script_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.getcwd(), TRAINING_SCRIPT),
                os.path.join(os.getcwd(), 'deployment', 'scripts', 'admin_training_script.py'),
                os.path.join(os.path.dirname(__file__), 'deployment', 'scripts', 'admin_training_script.py'),
                TRAINING_SCRIPT
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    script_path = path
                    break
        
        if not os.path.exists(script_path):
            error_msg = f"Training script not found: {script_path}"
            update_job_status(job_id, 'failed', error_msg)
            log_to_database(job_id, 'ERROR', error_msg)
            return
        
        # Run training script
        cmd = [
            'python', script_path,
            '--job_id', str(job_id),
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
        
        log_to_database(job_id, 'INFO', f'Running: {" ".join(cmd)}')
        log_to_database(job_id, 'INFO', f'Training script found at: {script_path}')
        
        # Run training with real-time output capture
        # Use Popen to capture output in real-time and log it
        import subprocess as sp
        process = sp.Popen(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line and log to database
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[Training] {line}")  # Also print to Heroku logs
                output_lines.append(line)
                # Log important lines to database
                if any(keyword in line.lower() for keyword in ['epoch', 'batch', 'loss', 'acc', 'saved', 'error', 'completed']):
                    log_to_database(job_id, 'INFO', line[:500])  # Limit length
        
        process.wait()
        returncode = process.returncode
        
        if returncode == 0:
            update_job_status(job_id, 'completed')
            log_to_database(job_id, 'INFO', 'Training completed successfully')
            # Log final summary
            final_lines = [l for l in output_lines[-10:] if l]  # Last 10 lines
            if final_lines:
                log_to_database(job_id, 'INFO', f'Final output: {" | ".join(final_lines)}')
        else:
            error_msg = '\n'.join(output_lines[-20:])[:500]  # Last 20 lines
            update_job_status(job_id, 'failed', error_msg)
            log_to_database(job_id, 'ERROR', f'Training failed (exit code {returncode}): {error_msg}')
            
    except subprocess.TimeoutExpired:
        error_msg = "Training timeout (exceeded 1 hour)"
        update_job_status(job_id, 'failed', error_msg)
        log_to_database(job_id, 'ERROR', error_msg)
    except Exception as e:
        error_msg = str(e)[:500]
        update_job_status(job_id, 'failed', error_msg)
        log_to_database(job_id, 'ERROR', f'Training error: {error_msg}')

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
def index() -> Any:
    """Root endpoint - API information"""
    return jsonify({
        "name": "AgriShield Combined API",
        "version": "1.0.0",
        "services": {
            "detection": {
                "framework": "ONNX Runtime",
                "status": "running" if ONNX_AVAILABLE and session else "error",
                "model_loaded": session is not None,
                "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none"
            },
            "training": {
                "status": "available",
                "database": "via PHP API Gateway",
                "method": "No direct MySQL access needed"
            }
        },
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "train": "/train (POST)",
            "training_status": "/status/<job_id> (GET)"
        }
    })

@app.get("/health")
def health() -> Any:
    """Health check endpoint"""
    detection_ok = ONNX_AVAILABLE and session is not None
    
    # Test PHP API connection (instead of direct database)
    api_ok = False
    api_error = None
    try:
        # Test PHP API by calling get_job with a test ID (should return 404, but confirms API is reachable)
        url = f"{PHP_API_BASE}/get_job.php"
        response = requests.get(url, params={'job_id': 999999}, timeout=5)
        # 404 is OK (job doesn't exist), 400 is OK (missing param), but 500 or connection error is bad
        if response.status_code in [200, 400, 404]:
            api_ok = True
        else:
            api_error = f"PHP API returned HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        api_error = f"PHP API unreachable: {str(e)}"
        print(f"PHP API connection error: {api_error}")
    
    return jsonify({
        "status": "ok" if detection_ok and api_ok else "partial",
        "detection": {
            "status": "ok" if detection_ok else "error",
            "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none"
        },
        "training": {
            "status": "ok" if api_ok else "error",
            "database": "connected" if api_ok else "disconnected",
            "method": "PHP API Gateway",
            "error": api_error if api_error else None
        }
    })

# ============================================================================
# PEST DETECTION ROUTES
# ============================================================================

def preprocess_image(image: Image.Image, input_shape: tuple) -> np.ndarray:
    """Preprocess image for ONNX model (supports both classification and YOLO)"""
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW format
            h, w = input_shape[2], input_shape[3]
        else:  # NHWC format
            h, w = input_shape[1], input_shape[2]
    else:
        # Default: Check if model is YOLO (640x640) or classification (224x224 or 512x512)
        # Try to detect from model path or use 640 for YOLO
        if ONNX_MODEL_PATH and 'yolo' in ONNX_MODEL_PATH.lower():
            h, w = 640, 640  # YOLO standard size
        else:
            h, w = 512, 512  # Default for classification
    
    # Resize maintaining aspect ratio (YOLO style)
    img_ratio = min(w / image.width, h / image.height)
    new_w, new_h = int(image.width * img_ratio), int(image.height * img_ratio)
    img_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create padded image
    img_padded = Image.new('RGB', (w, h), (114, 114, 114))  # Gray padding
    img_padded.paste(img_resized, ((w - new_w) // 2, (h - new_h) // 2))
    
    img_array = np.array(img_padded, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    if len(img_array.shape) == 3:  # HWC
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def postprocess_output(output_data: np.ndarray, conf_threshold: float = None) -> Dict[str, int]:
    """Postprocess ONNX model output to get pest counts (supports both classification and YOLO)"""
    if conf_threshold is None:
        conf_threshold = DETECTION_CONF_THRESHOLD
    """Postprocess ONNX model output to get pest counts (supports both classification and YOLO)"""
    counts = {name: 0 for name in CLASS_NAMES}
    
    if len(CLASS_NAMES) == 0:
        print("⚠️  No class names defined!")
        return counts
    
    print(f"🔍 Postprocessing output: shape={output_data.shape}, classes={len(CLASS_NAMES)}, threshold={conf_threshold}")
    
    # Handle different output shapes
    if len(output_data.shape) == 3:
        # Shape could be [batch, features, num_detections] or [batch, num_detections, features]
        output_data = output_data[0]  # Remove batch dimension
        
        # Check if it's [features, num_detections] format (transpose needed)
        # YOLO models often output [batch, features, num_detections]
        # With shape [9, 5376], 9 is likely features, 5376 is num_detections
        if output_data.shape[0] < output_data.shape[1] and output_data.shape[0] <= 20:
            # Likely [features, num_detections] - transpose to [num_detections, features]
            print(f"🔍 Transposing output from {output_data.shape} to {output_data.T.shape} (detected [features, detections] format)")
            detections = output_data.T
        else:
            # Likely [num_detections, features] - use as is
            print(f"🔍 Using output as-is: {output_data.shape} (detected [detections, features] format)")
            detections = output_data
    elif len(output_data.shape) == 2:
        # Shape: [num_detections, features]
        detections = output_data
    elif len(output_data.shape) == 4:
        # Shape: [1, classes, H, W] - Classification model output
        # Convert to detection format (not ideal, but for backward compatibility)
        print(f"🔍 Classification model detected: shape {output_data.shape}")
        output_data = output_data[0]  # Remove batch dimension
        if output_data.shape[0] == len(CLASS_NAMES):
            # Classification output: [classes, H, W] -> get max class
            class_probs = np.mean(output_data, axis=(1, 2))  # Average over spatial dimensions
            max_class = int(np.argmax(class_probs))
            max_conf = float(np.max(class_probs))
            
            # Get all class probabilities for debugging
            all_probs = {CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))}
            print(f"🔍 All class probabilities: {all_probs}")
            print(f"🔍 Max: class={max_class} ({CLASS_NAMES[max_class] if max_class < len(CLASS_NAMES) else 'unknown'}), confidence={max_conf:.4f}, threshold={conf_threshold}")
            
            # Higher threshold for classification to reduce false positives
            # Also check if max confidence is significantly higher than other classes
            min_conf_threshold = max(conf_threshold, CLASSIFICATION_MIN_THRESHOLD)  # At least configured minimum for classification
            second_max_conf = float(np.partition(class_probs, -2)[-2]) if len(class_probs) > 1 else 0
            confidence_gap = max_conf - second_max_conf
            
            print(f"🔍 Confidence check: max={max_conf:.4f}, second_max={second_max_conf:.4f}, gap={confidence_gap:.4f}, min_threshold={min_conf_threshold:.4f}")
            
            # Require: high confidence AND significant gap from other classes (reduces false positives)
            if max_conf >= min_conf_threshold and confidence_gap >= CONFIDENCE_GAP_REQUIREMENT and 0 <= max_class < len(CLASS_NAMES):
                counts[CLASS_NAMES[max_class]] = 1  # Classification: only 1 detection
                print(f"✅ Detection accepted: {CLASS_NAMES[max_class]} (conf={max_conf:.4f}, gap={confidence_gap:.4f})")
            else:
                if max_conf < min_conf_threshold:
                    print(f"⚠️  Detection rejected: confidence {max_conf:.4f} < minimum threshold {min_conf_threshold:.4f}")
                elif confidence_gap < CONFIDENCE_GAP_REQUIREMENT:
                    print(f"⚠️  Detection rejected: confidence gap too small ({confidence_gap:.4f} < {CONFIDENCE_GAP_REQUIREMENT}) - likely false positive")
                else:
                    print(f"⚠️  Detection rejected: class index out of range")
        else:
            print(f"⚠️  Class count mismatch: model has {output_data.shape[0]} classes, expected {len(CLASS_NAMES)}")
        return counts
    else:
        return counts
    
    # Process detections (YOLO format: [x, y, w, h, conf, class_id, ...] or [x1, y1, x2, y2, conf, class_id, ...])
    print(f"🔍 Processing {len(detections)} detections (YOLO format)")
    
    # Collect all valid detections with bounding boxes for NMS
    valid_detections = []
    all_detections_by_class = {i: [] for i in range(len(CLASS_NAMES))}  # Track all detections per class for debugging
    
    for i, detection in enumerate(detections):
        if len(detection) < 6:
            continue
        
        conf = None
        class_id = None
        bbox = None
        
        # Extract bbox and confidence
        if len(detection) >= 6:
            # Format: [x, y, w, h, conf, class_id]
            x, y, w, h = float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])
            conf = float(detection[4])
            class_id = int(detection[5])
            # Convert center format to corner format for NMS: [x1, y1, x2, y2]
            bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
        elif len(detection) >= 5:
            objectness = float(detection[4])
            if len(detection) > 5:
                class_probs = np.array(detection[5:])
                max_class_idx = int(np.argmax(class_probs))
                max_class_prob = float(class_probs[max_class_idx])
                conf = objectness * max_class_prob
                class_id = max_class_idx
                x, y, w, h = float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])
                bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
            else:
                continue
        
        if conf is not None and class_id is not None and bbox is not None:
            # Track all detections for debugging
            if 0 <= class_id < len(CLASS_NAMES):
                all_detections_by_class[class_id].append(conf)
            
            yolo_threshold = max(conf_threshold, YOLO_CONF_THRESHOLD)
            if conf >= yolo_threshold and 0 <= class_id < len(CLASS_NAMES):
                valid_detections.append({
                    'bbox': bbox,
                    'conf': conf,
                    'class_id': class_id
                })
            elif 0 <= class_id < len(CLASS_NAMES) and i < 10:  # Log first 10 rejected per class
                print(f"⚠️  Rejected {CLASS_NAMES[class_id]}: conf={conf:.4f} < threshold {yolo_threshold:.4f}")
    
    # Debug: Show detection statistics per class
    print(f"🔍 Detection statistics per class:")
    for class_id in range(len(CLASS_NAMES)):
        all_confs = all_detections_by_class[class_id]
        if len(all_confs) > 0:
            max_conf = max(all_confs)
            avg_conf = sum(all_confs) / len(all_confs)
            above_threshold = sum(1 for c in all_confs if c >= max(conf_threshold, YOLO_CONF_THRESHOLD))
            print(f"   {CLASS_NAMES[class_id]}: {len(all_confs)} total, max={max_conf:.4f}, avg={avg_conf:.4f}, above_threshold={above_threshold}")
    
    # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
    if len(valid_detections) > 0:
        print(f"🔍 Found {len(valid_detections)} valid detections before NMS")
        
        # Group by class and apply NMS per class
        nms_detections = []
        for class_id in range(len(CLASS_NAMES)):
            class_dets = [d for d in valid_detections if d['class_id'] == class_id]
            if len(class_dets) == 0:
                continue
            
            # Sort by confidence (highest first)
            class_dets.sort(key=lambda x: x['conf'], reverse=True)
            
            # Simple NMS: keep highest confidence, remove overlapping boxes
            kept = []
            for det in class_dets:
                overlap = False
                for kept_det in kept:
                    # Calculate IoU (Intersection over Union)
                    iou = calculate_iou(det['bbox'], kept_det['bbox'])
                    if iou > 0.5:  # 50% overlap threshold
                        overlap = True
                        break
                if not overlap:
                    kept.append(det)
            
            nms_detections.extend(kept)
            if len(class_dets) > len(kept):
                print(f"🔍 NMS: {len(class_dets)} -> {len(kept)} detections for {CLASS_NAMES[class_id]}")
        
        valid_detections = nms_detections
        print(f"🔍 After NMS: {len(valid_detections)} unique detections")
    
    # Count detections after NMS
    detection_count = 0
    for det in valid_detections:
        counts[CLASS_NAMES[det['class_id']]] += 1
        detection_count += 1
        if detection_count <= 3:
            print(f"✅ Accepted: {CLASS_NAMES[det['class_id']]} (conf={det['conf']:.4f})")
    
    print(f"🔍 Total detections accepted: {detection_count}")
    
    return counts

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # Box format: [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

# Old code removed - keeping for reference but replaced above
def _old_detection_processing():
    """Old detection processing code - replaced by NMS version"""
    detection_count = 0
    for i, detection in enumerate(detections):
        if len(detection) < 6:
            continue
        
        # YOLO output format can vary:
        # Format 1: [x_center, y_center, width, height, confidence, class_id, ...]
        # Format 2: [x1, y1, x2, y2, confidence, class_id, ...]
        # Format 3: [batch, x, y, w, h, conf, class_id, ...] (if batch dimension present)
        
        # Try to find confidence and class_id
        # Usually at indices 4 and 5, but could be different
        conf = None
        class_id = None
        
        # Check if it's YOLO format (has bounding box coordinates)
        # YOLO output format: [x, y, w, h, objectness, class_prob_0, class_prob_1, ...]
        # Or: [x, y, w, h, conf, class_id] (if class_id is already determined)
        # Or: [x, y, w, h, class_conf_0, class_conf_1, ...] (class-specific confidences)
        
        if len(detection) >= 6:
            # Try standard format: [x, y, w, h, conf, class_id]
            conf = float(detection[4])
            class_id = int(detection[5])
        elif len(detection) >= 5:
            # Format: [x, y, w, h, objectness, class_probs...]
            # Or: [x, y, w, h, conf] with class_id elsewhere
            objectness = float(detection[4])
            
            # If we have more than 5 values, check if they're class probabilities
            if len(detection) > 5:
                # Extract class probabilities (indices 5 onwards)
                class_probs = np.array(detection[5:])
                max_class_idx = int(np.argmax(class_probs))
                max_class_prob = float(class_probs[max_class_idx])
                
                # Combined confidence = objectness * class_probability
                conf = objectness * max_class_prob
                class_id = max_class_idx
            else:
                # Just objectness, no class info - skip
                continue
        
        if conf is not None and class_id is not None:
            if i < 3:  # Log first 3 detections for debugging
                print(f"🔍 Detection {i}: class_id={class_id}, conf={conf:.4f}, threshold={conf_threshold}")
            
            # For YOLO, use higher threshold to reduce false positives
            yolo_threshold = max(conf_threshold, YOLO_CONF_THRESHOLD)
            
            if conf >= yolo_threshold and 0 <= class_id < len(CLASS_NAMES):
                counts[CLASS_NAMES[class_id]] += 1
                detection_count += 1
                if detection_count <= 3:
                    print(f"✅ Accepted: {CLASS_NAMES[class_id]} (conf={conf:.4f} >= {yolo_threshold})")
            elif i < 3:
                if conf < yolo_threshold:
                    print(f"⚠️  Rejected: conf={conf:.4f} < threshold {yolo_threshold}")
                elif class_id < 0 or class_id >= len(CLASS_NAMES):
                    print(f"⚠️  Rejected: class_id={class_id} out of range [0, {len(CLASS_NAMES)})")
    
    print(f"🔍 Total detections accepted: {detection_count}")
    
    return counts

@app.post("/detect")
def detect() -> Any:
    """Pest detection endpoint using ONNX Runtime"""
    if not ONNX_AVAILABLE or session is None:
        return jsonify({
            "error": "ONNX model not available",
            "message": "Install ONNX Runtime or load model"
        }), 500
    
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400
    
    t0 = time.time()
    
    # Preprocess
    # Get input shape from model, default to 640x640 for YOLO or 512x512 for classification
    if input_details.shape:
        input_shape = input_details.shape
    else:
        # Try to detect model type from path
        if ONNX_MODEL_PATH and ('yolo' in ONNX_MODEL_PATH.lower() or 'job_' in ONNX_MODEL_PATH):
            input_shape = [1, 3, 640, 640]  # YOLO standard
        else:
            input_shape = [1, 3, 512, 512]  # Classification default
    
    input_data = preprocess_image(img, tuple(input_shape))
    
    # Run inference
    input_name = input_details.name
    output = session.run([output_details.name], {input_name: input_data})
    output_data = output[0]
    
    dt = time.time() - t0
    
    # Debug: Log model output for troubleshooting
    print(f"🔍 Model output shape: {output_data.shape}")
    print(f"🔍 Model output min/max: {output_data.min():.4f} / {output_data.max():.4f}")
    print(f"🔍 Model output mean: {output_data.mean():.4f}")
    if len(output_data.shape) >= 2:
        print(f"🔍 Model output sample (first 5 values): {output_data.flatten()[:5]}")
    
    # Postprocess with configured threshold
    counts = postprocess_output(output_data, conf_threshold=DETECTION_CONF_THRESHOLD)
    total_detections = sum(counts.values())
    
    # Don't lower threshold automatically - this causes false positives
    # Instead, keep the threshold high to reduce false positives
    
    # Debug: Log detection results
    print(f"🔍 Final detection results: {total_detections} total pests detected")
    for pest, count in counts.items():
        if count > 0:
            print(f"   ✅ {pest}: {count}")
        else:
            print(f"   ❌ {pest}: 0")
    
    # Pesticide recommendations
    pesticide_recs = {
        "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
        "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
        "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
        "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
    }
    
    recommendations = {k: v for k, v in pesticide_recs.items() if counts.get(k, 0) > 0}
    
    # Verification System: Determine if detected pests are known/verified
    verified_pests = {}
    unverified_detections = []
    
    # Check each detected pest
    for pest_name, count in counts.items():
        if count > 0:
            # If pest is in CLASS_NAMES, it's a known/verified pest
            if pest_name in CLASS_NAMES:
                verified_pests[pest_name] = count
            else:
                # Unknown pest detected
                unverified_detections.append({
                    "pest_name": pest_name,
                    "count": count,
                    "reason": "not_in_training_data"
                })
    
    # Determine overall verification status
    has_verified = len(verified_pests) > 0
    has_unverified = len(unverified_detections) > 0
    
    if has_verified and not has_unverified:
        verification_status = "verified"  # All detections are known pests
    elif has_unverified and not has_verified:
        verification_status = "unverified"  # Only unknown detections
    elif has_verified and has_unverified:
        verification_status = "mixed"  # Both known and unknown
    else:
        verification_status = "no_detection"  # No pests detected
    
    return jsonify({
        "status": "success",
        "pest_counts": counts,
        "verified_pests": verified_pests,
        "unverified_detections": unverified_detections,
        "verification_status": verification_status,
        "is_known_pest": has_verified,  # True if at least one known pest detected
        "requires_manual_review": has_unverified,  # True if unknown pests detected
        "recommendations": recommendations,
        "inference_time_ms": round(dt * 1000, 1),
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none",
        "framework": "ONNX Runtime",
        "known_classes": CLASS_NAMES  # List of all known pest classes
    })

# ============================================================================
# TRAINING ROUTES
# ============================================================================

@app.route('/train', methods=['POST'])
def start_training():
    """Start training job"""
    try:
        data = request.json or {}
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'success': False, 'message': 'job_id required'}), 400
        
        # Check if job exists and is pending
        job = get_training_job(job_id)
        if not job:
            return jsonify({'success': False, 'message': 'Job not found'}), 404
        
        if job['status'] != 'pending':
            return jsonify({
                'success': False, 
                'message': f'Job status is {job["status"]}, expected pending'
            }), 400
        
        # Start training in background thread
        thread = threading.Thread(target=run_training, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'job_id': job_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/status/<int:job_id>', methods=['GET'])
def get_status(job_id):
    """Get training job status"""
    try:
        job = get_training_job(job_id)
        if not job:
            return jsonify({'success': False, 'message': 'Job not found'}), 404
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': job['status'],
            'completed_at': job.get('completed_at'),
            'error_message': job.get('error_message')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"Starting Combined AgriShield API on port {port}...")
    print(f"  - Pest Detection: ONNX Runtime")
    print(f"  - Training Service: PyTorch")
    app.run(host="0.0.0.0", port=port, debug=False)

