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

def find_onnx_model() -> str:
    """Find ONNX model file"""
    base_dir = Path(__file__).resolve().parent
    
    # Priority order - check models/ directory first (for Heroku deployment)
    candidates = [
        base_dir / "models" / "best 2.onnx",
        base_dir / "models" / "best.onnx",
        base_dir / "models" / "best5.onnx",
        base_dir / "deployment" / "models" / "best 2.onnx",
        base_dir / "deployment" / "models" / "best.onnx",
        base_dir / "deployment" / "models" / "best5.onnx",
        base_dir / "datasets" / "best 2.onnx",
        base_dir / "datasets" / "best.onnx",
        base_dir / "datasets" / "best5.onnx",
        base_dir / "pest_detection_ml" / "models" / "best.onnx",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    raise FileNotFoundError(
        f"ONNX model not found. Checked: {[str(c) for c in candidates]}"
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
        session, input_details, output_details = load_onnx_model(ONNX_MODEL_PATH)
        
        # Default class names (update based on your model)
        CLASS_NAMES = [
            "Rice_Bug",
            "White stem borer",
            "black-bug",
            "brown_hopper",
            "green_hopper",
        ]
    except Exception as e:
        print(f"⚠️  Could not load ONNX model: {e}")
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
        
        config = json.loads(job.get('training_config', '{}'))
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            update_job_status(job_id, 'completed')
            log_to_database(job_id, 'INFO', 'Training completed successfully')
        else:
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            update_job_status(job_id, 'failed', error_msg)
            log_to_database(job_id, 'ERROR', f'Training failed: {error_msg}')
            
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
    """Preprocess image for ONNX model"""
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW format
            h, w = input_shape[2], input_shape[3]
        else:  # NHWC format
            h, w = input_shape[1], input_shape[2]
    else:
        h, w = 512, 512  # Default
    
    img = image.resize((w, h))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    
    if len(img_array.shape) == 3:  # HWC
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def postprocess_output(output_data: np.ndarray, conf_threshold: float = 0.15) -> Dict[str, int]:
    """Postprocess ONNX model output to get pest counts"""
    counts = {name: 0 for name in CLASS_NAMES}
    
    if len(output_data.shape) == 3:
        detections = output_data[0]
    elif len(output_data.shape) == 2:
        detections = output_data
    else:
        return counts
    
    for detection in detections:
        if len(detection) >= 6:
            conf = float(detection[4])
            class_id = int(detection[5])
            
            if conf >= conf_threshold and 0 <= class_id < len(CLASS_NAMES):
                counts[CLASS_NAMES[class_id]] += 1
    
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
    input_shape = input_details.shape if input_details.shape else [1, 3, 512, 512]
    input_data = preprocess_image(img, input_shape)
    
    # Run inference
    input_name = input_details.name
    output = session.run([output_details.name], {input_name: input_data})
    output_data = output[0]
    
    dt = time.time() - t0
    
    # Postprocess
    counts = postprocess_output(output_data, conf_threshold=0.15)
    
    # Pesticide recommendations
    pesticide_recs = {
        "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
        "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
        "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
        "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
    }
    
    recommendations = {k: v for k, v in pesticide_recs.items() if counts.get(k, 0) > 0}
    
    return jsonify({
        "status": "success",
        "pest_counts": counts,
        "recommendations": recommendations,
        "inference_time_ms": round(dt * 1000, 1),
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none",
        "framework": "ONNX Runtime"
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
