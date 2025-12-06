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
MODEL_VERSION = None  # Store model version from server
MODEL_ACCURACY = None  # Store model accuracy from server

# ============================================================================
# MODEL CACHING SYSTEM (Farm-Specific Models)
# ============================================================================

class ModelCache:
    """Cache for farm-specific models with version tracking"""
    
    def __init__(self):
        self.cache = {}  # {farm_id: {'session': session, 'input_details': input_details, 'output_details': output_details, 'model_path': path, 'version': version, 'accuracy': accuracy, 'last_used': timestamp}}
        self.lock = threading.Lock()
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, farm_id=None, device_id=None):
        """Generate cache key from farm_id or device_id"""
        if farm_id:
            return f"farm_{farm_id}"
        elif device_id:
            return f"device_{device_id}"
        return "global"
    
    def get_cached_model(self, farm_id=None, device_id=None):
        """Get cached model if available and valid"""
        cache_key = self.get_cache_key(farm_id, device_id)
        
        with self.lock:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                # Check if model file still exists
                if cached['model_path'] and Path(cached['model_path']).exists():
                    cached['last_used'] = time.time()
                    return cached
                else:
                    # Model file deleted, remove from cache
                    del self.cache[cache_key]
        
        return None
    
    def cache_model(self, farm_id=None, device_id=None, session=None, input_details=None, 
                   output_details=None, model_path=None, version=None, accuracy=None):
        """Cache a loaded model"""
        cache_key = self.get_cache_key(farm_id, device_id)
        
        with self.lock:
            self.cache[cache_key] = {
                'session': session,
                'input_details': input_details,
                'output_details': output_details,
                'model_path': model_path,
                'version': version,
                'accuracy': accuracy,
                'last_used': time.time()
            }
            print(f"✅ Cached model for {cache_key}: {version} (accuracy: {accuracy}%)")
    
    def check_model_version(self, farm_id=None, device_id=None, current_version=None):
        """Check if cached model version matches current version"""
        cached = self.get_cached_model(farm_id, device_id)
        if cached and cached.get('version') == current_version:
            return True
        return False
    
    def invalidate_cache(self, farm_id=None, device_id=None):
        """Invalidate cache for specific farm/device"""
        cache_key = self.get_cache_key(farm_id, device_id)
        with self.lock:
            if cache_key in self.cache:
                # Close session if exists
                if self.cache[cache_key].get('session'):
                    try:
                        self.cache[cache_key]['session'].close()
                    except:
                        pass
                del self.cache[cache_key]
                print(f"🗑️  Invalidated cache for {cache_key}")
    
    def cleanup_old_models(self, max_age_seconds=3600):
        """Remove models from cache that haven't been used recently"""
        current_time = time.time()
        with self.lock:
            to_remove = []
            for key, cached in self.cache.items():
                if current_time - cached.get('last_used', 0) > max_age_seconds:
                    to_remove.append(key)
            
            for key in to_remove:
                if self.cache[key].get('session'):
                    try:
                        self.cache[key]['session'].close()
                    except:
                        pass
                del self.cache[key]
                print(f"🧹 Cleaned up unused model cache: {key}")

# Global model cache instance
model_cache = ModelCache()

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
    """Find ONNX model file - ALWAYS checks server first, then uses local files as fallback"""
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # FIRST: ALWAYS try to download latest model from server
    print("🔍 Checking server for latest model...")
    try:
        web_server_url = os.getenv('WEB_SERVER_URL', 'https://agrishield.bccbsis.com/Proto1')
        
        # Try to get model info first (optional - if it fails, still try download)
        model_info_url = f"{web_server_url}/api/training/get_active_model_info.php"
        server_version = None
        server_accuracy = None
        
        try:
            info_response = requests.get(model_info_url, timeout=5)
            if info_response.status_code == 200:
                model_info = info_response.json()
                server_version = model_info.get('version')
                server_accuracy = model_info.get('accuracy')
                print(f"📊 Server has active model: {server_version} (accuracy: {server_accuracy}%)")
            elif info_response.status_code == 404:
                # 404 means no active model in database, but model file might still exist
                print(f"   ℹ️  No active model in database (404), but trying download endpoint anyway...")
            else:
                print(f"   ⚠️  Model info check failed: HTTP {info_response.status_code}, trying download endpoint...")
        except Exception as e:
            print(f"   ⚠️  Model info check error: {e}, trying download endpoint...")
        
        # Always try to download (even if info check failed)
        print(f"📥 Attempting to download latest model from server...")
        download_url = f"{web_server_url}/api/training/get_active_model.php"
        response = requests.get(download_url, timeout=180, stream=True)  # 3 minutes for 42MB
        
        if response.status_code == 200:
            downloaded_model_path = models_dir / "best.onnx"
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(downloaded_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (5 * 1024 * 1024) == 0:  # Print every 5MB
                            progress = (downloaded / total_size) * 100
                            print(f"   Downloaded: {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({progress:.1f}%)")
            
            # Get version from headers or use from info check
            model_version = response.headers.get('X-Model-Version', server_version or 'N/A')
            model_accuracy = response.headers.get('X-Model-Accuracy', str(server_accuracy) if server_accuracy else 'N/A')
            
            file_size = downloaded_model_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"   ✅ Model downloaded: {model_version} (accuracy: {model_accuracy}%) - {file_size:.2f} MB")
            
            # Copy to best 2.onnx to replace the old default model
            best2_path = models_dir / "best 2.onnx"
            import shutil
            shutil.copy2(downloaded_model_path, best2_path)
            print(f"   📋 Copied latest model to best 2.onnx (replacing old default)")
            
            # Store model metadata globally
            global MODEL_VERSION, MODEL_ACCURACY
            MODEL_VERSION = model_version if model_version != 'N/A' else None
            MODEL_ACCURACY = float(model_accuracy) if model_accuracy != 'N/A' else None
            
            return str(downloaded_model_path)
        elif response.status_code == 404:
            print(f"   ⚠️  No model available on server (404)")
            print(f"   Will use local models if available")
        else:
            print(f"   ⚠️  Download failed: HTTP {response.status_code}")
            print(f"   Will use local models if available")
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Server check timed out (will use local models)")
    except Exception as e:
        import traceback
        print(f"   ⚠️  Server check failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print("   Will use local models if available")
    
    # FALLBACK: Check local files (fast, no network dependency)
    print("🔍 Checking local model files...")
    
    # IMPORTANT: Check models/best.onnx first (downloaded models have highest priority)
    downloaded_check = models_dir / "best.onnx"
    if downloaded_check.exists():
        file_size = downloaded_check.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"📦 Found downloaded model in fallback: best.onnx ({file_size:.2f} MB)")
        return str(downloaded_check)
    
    candidates = [
        base_dir / "models" / "best 2.onnx",  # Default model from git (lower priority)
        base_dir / "models" / "best5.onnx",
        base_dir / "deployment" / "models" / "best.onnx",
        base_dir / "deployment" / "models" / "best 2.onnx",
        base_dir / "deployment" / "models" / "best5.onnx",
        base_dir / "datasets" / "best.onnx",
        base_dir / "datasets" / "best 2.onnx",
        base_dir / "datasets" / "best5.onnx",
        base_dir / "pest_detection_ml" / "models" / "best.onnx",
    ]
    
    # Check standard locations
    for candidate in candidates:
        if candidate.exists():
            file_size = candidate.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"📦 Found local model: {candidate.name} ({file_size:.2f} MB)")
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
    
    
    # If still not found, raise error
    raise FileNotFoundError(
        f"ONNX model not found. Checked local files and job directories. "
        f"Note: On Heroku, models are automatically downloaded from server if no local model exists."
    )

def _check_server_for_updates_async(base_dir: Path):
    """Check server for model updates in background (non-blocking)"""
    def check_update():
        try:
            web_server_url = os.getenv('WEB_SERVER_URL', 'https://agrishield.bccbsis.com/Proto1')
            model_info_url = f"{web_server_url}/api/training/get_active_model_info.php"
            info_response = requests.get(model_info_url, timeout=5)
            
            if info_response.status_code == 200:
                model_info = info_response.json()
                server_version = model_info.get('version')
                server_accuracy = model_info.get('accuracy')
                
                print(f"📊 Server has newer model: {server_version} (accuracy: {server_accuracy}%)")
                print("   (Will download in background)")
                
                # Download in background
                download_url = f"{web_server_url}/api/training/get_active_model.php"
                response = requests.get(download_url, timeout=60, stream=True)
                
                if response.status_code == 200:
                    models_dir = base_dir / "models"
                    models_dir.mkdir(exist_ok=True)
                    downloaded_model_path = models_dir / "best.onnx"
                    
                    with open(downloaded_model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    print(f"   ✅ Updated model downloaded: {server_version}")
                    
                    # Copy to best 2.onnx to replace the old default model
                    best2_path = models_dir / "best 2.onnx"
                    import shutil
                    shutil.copy2(downloaded_model_path, best2_path)
                    print(f"   📋 Updated best 2.onnx with latest model")
                    
                    global MODEL_VERSION, MODEL_ACCURACY
                    MODEL_VERSION = server_version
                    MODEL_ACCURACY = server_accuracy
        except Exception:
            pass  # Silently fail - don't block app startup
    
    # Start background thread
    thread = threading.Thread(target=check_update, daemon=True)
    thread.start()

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

# Initialize model (with automatic download from server if needed)
if ONNX_AVAILABLE:
    try:
        ONNX_MODEL_PATH = find_onnx_model()
        model_name = Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none"
        print(f"🔍 Found model at: {ONNX_MODEL_PATH}")
        print(f"📋 Model filename: {model_name}")
        
        # Try to get class names and model metadata from server (optional - don't fail if this fails)
        try:
            web_server_url = os.getenv('WEB_SERVER_URL', 'https://agrishield.bccbsis.com/Proto1')
            model_info_url = f"{web_server_url}/api/training/get_active_model_info.php"
            info_response = requests.get(model_info_url, timeout=5)  # Shorter timeout
            if info_response.status_code == 200:
                model_info = info_response.json()
                if 'classes' in model_info and model_info['classes']:
                    CLASS_NAMES[:] = model_info['classes']  # Update list in place
                    print(f"📋 Loaded class names from server: {CLASS_NAMES}")
                
                # Store model metadata (module-level variables, no global needed)
                if 'version' in model_info:
                    MODEL_VERSION = model_info['version']
                if 'accuracy' in model_info:
                    MODEL_ACCURACY = float(model_info['accuracy'])  # Convert to float
                    print(f"📊 Model version: {MODEL_VERSION}, accuracy: {MODEL_ACCURACY*100:.1f}%")
        except Exception as e:
            # Don't fail app startup if this fails - just use defaults
            print(f"⚠️  Could not load class names from server: {e}")
            print("   Using default class names")
        
        # Set default class names only if not loaded from server
        if not CLASS_NAMES:
            CLASS_NAMES = [
                "Rice_Bug",
                "White stem borer",
                "black-bug",
                "brown_hopper",
                "green_hopper",
        ]
        
        session, input_details, output_details = load_onnx_model(ONNX_MODEL_PATH)
        
        print(f"✅ Model loaded successfully: {Path(ONNX_MODEL_PATH).name}")
        print(f"   Classes: {CLASS_NAMES}")
    except FileNotFoundError as e:
        # Model file not found - this is OK, app can still run (detection will fail but won't crash)
        import traceback
        print(f"⚠️  ONNX model file not found: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print("   App will start but detection will be unavailable until a model is provided")
        session = None
        input_details = None
        output_details = None
        ONNX_MODEL_PATH = None
    except Exception as e:
        # Any other error - log it but don't crash the app
        import traceback
        print(f"⚠️  Could not load ONNX model: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        print("   App will start but detection will be unavailable")
        session = None
        input_details = None
        output_details = None
        ONNX_MODEL_PATH = None
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
        
        # PREFERRED: Use dedicated columns if they exist (more reliable)
        epochs = job.get('epochs')
        batch_size = job.get('batch_size')
        
        # If columns don't exist or are None, fall back to training_config JSON
        if epochs is None or batch_size is None:
            print(f"[INFO] epochs/batch_size columns not found, falling back to training_config JSON")
            training_config_raw = job.get('training_config', '{}')
            
            # IMPORTANT: Log the raw config first
            print(f"[DEBUG] Raw training_config from database: {training_config_raw} (type: {type(training_config_raw)})")
            
            if isinstance(training_config_raw, str):
                try:
                    if training_config_raw and training_config_raw.strip() and training_config_raw.strip() != '{}':
                        config = json.loads(training_config_raw)
                    else:
                        print(f"[WARNING] training_config is empty string or '{{}}', using defaults")
                        config = {}
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON in training_config: {training_config_raw}")
                    print(f"[ERROR] JSON decode error: {e}")
                    config = {}
            elif isinstance(training_config_raw, dict):
                config = training_config_raw
            else:
                config = {}
            
            # Extract from config if columns not available
            if epochs is None:
                epochs = config.get('epochs') if isinstance(config, dict) else None
            if batch_size is None:
                batch_size = config.get('batch_size') if isinstance(config, dict) else None
        
        # Convert to int and use defaults if not set
        try:
            epochs = int(epochs) if epochs is not None and epochs != '' else 50
        except (ValueError, TypeError):
            print(f"[WARNING] epochs value '{epochs}' is not a valid integer, using default 50")
            epochs = 50
        
        try:
            batch_size = int(batch_size) if batch_size is not None and batch_size != '' else 8
        except (ValueError, TypeError):
            print(f"[WARNING] batch_size value '{batch_size}' is not a valid integer, using default 8")
            batch_size = 8
        
        # IMPORTANT: Log the config parsing for debugging
        print(f"[DEBUG] Training config parsed: {config}")
        print(f"[DEBUG] Epochs value: {epochs} (type: {type(epochs)})")
        print(f"[DEBUG] Batch size value: {batch_size} (type: {type(batch_size)})")
        print(f"[INFO] Using training config: epochs={epochs}, batch_size={batch_size}")
        
        # Log to database for visibility
        log_to_database(job_id, 'INFO', f'Training config parsed: epochs={epochs}, batch_size={batch_size} (from config: {config})')
        
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
        
        # Ensure epochs is an integer
        epochs = int(epochs) if epochs else 50
        batch_size = int(batch_size) if batch_size else 8
        
        # Run training script
        cmd = [
            'python', script_path,
            '--job_id', str(job_id),
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
        
        print(f"[DEBUG] Command being executed: {' '.join(cmd)}")
        log_to_database(job_id, 'INFO', f'Running: {" ".join(cmd)}')
        log_to_database(job_id, 'INFO', f'Starting training with epochs={epochs}, batch_size={batch_size}')
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
    
    detection_info = {
        "status": "ok" if detection_ok else "error",
        "model": Path(ONNX_MODEL_PATH).name if ONNX_MODEL_PATH else "none"
    }
    
    # Add model metadata if available
    if MODEL_VERSION:
        detection_info["model_version"] = MODEL_VERSION
    if MODEL_ACCURACY:
        detection_info["model_accuracy"] = float(MODEL_ACCURACY)
    
    return jsonify({
        "status": "ok" if detection_ok and api_ok else "partial",
        "detection": detection_info,
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
        # OR classification: [batch, classes, spatial] like [1, 9, 5376]
        output_data = output_data[0]  # Remove batch dimension
        
        # Check if it's classification format: [classes, spatial_locations]
        # For ResNet18 classification, if first dim matches num_classes, it's classification
        if output_data.shape[0] == len(CLASS_NAMES) and output_data.shape[0] < 20:
            # Classification model: [classes, spatial] -> average over spatial dims
            print(f"🔍 Classification model detected: shape {output_data.shape} (classes={output_data.shape[0]})")
            class_probs = np.mean(output_data, axis=1)  # Average over spatial dimension
            max_class = int(np.argmax(class_probs))
            max_conf = float(np.max(class_probs))
            
            all_probs = {CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))}
            print(f"🔍 All class probabilities: {all_probs}")
            print(f"🔍 Max: class={max_class} ({CLASS_NAMES[max_class] if max_class < len(CLASS_NAMES) else 'unknown'}), confidence={max_conf:.4f}, threshold={conf_threshold}")
            
            min_conf_threshold = max(conf_threshold, CLASSIFICATION_MIN_THRESHOLD)
            second_max_conf = float(np.partition(class_probs, -2)[-2]) if len(class_probs) > 1 else 0
            confidence_gap = max_conf - second_max_conf
            
            if max_conf >= min_conf_threshold and confidence_gap >= CONFIDENCE_GAP_REQUIREMENT and 0 <= max_class < len(CLASS_NAMES):
                counts[CLASS_NAMES[max_class]] = 1
                print(f"✅ Detection accepted: {CLASS_NAMES[max_class]} (conf={max_conf:.4f}, gap={confidence_gap:.4f})")
            else:
                print(f"⚠️  Detection rejected: conf={max_conf:.4f} < {min_conf_threshold:.4f} or gap={confidence_gap:.4f} < {CONFIDENCE_GAP_REQUIREMENT}")
            return counts
        
        # Check if it's [features, num_detections] format (YOLO - transpose needed)
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
    """Pest detection endpoint using ONNX Runtime - supports farm/device-specific models"""
    global session, input_details, output_details, ONNX_MODEL_PATH, MODEL_VERSION, MODEL_ACCURACY
    
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
    
    # Get farm_id or device_id from form data (optional - for farm-specific models)
    farm_id = request.form.get('farm_id') or request.form.get('farm_parcels_id')
    device_id = request.form.get('device_id')
    
    # Track which model is being used for logging
    model_source = "global"
    used_farm_id = None
    
    # If device_id provided, try to get farm-specific model
    if device_id or farm_id:
        try:
            # First, check cache for existing model
            cached_model = model_cache.get_cached_model(farm_id=farm_id, device_id=device_id)
            
            if cached_model:
                # Use cached model
                session = cached_model['session']
                input_details = cached_model['input_details']
                output_details = cached_model['output_details']
                ONNX_MODEL_PATH = cached_model['model_path']
                MODEL_VERSION = cached_model.get('version', 'N/A')
                MODEL_ACCURACY = cached_model.get('accuracy', 'N/A')
                model_source = f"cached_{model_cache.get_cache_key(farm_id, device_id)}"
                used_farm_id = farm_id or device_id
                print(f"✅ Using cached model for {model_cache.get_cache_key(farm_id, device_id)}: {MODEL_VERSION}")
            else:
                # Download and cache new model
                web_server_url = os.getenv('WEB_SERVER_URL', 'https://agrishield.bccbsis.com/Proto1')
                model_url = f"{web_server_url}/api/training/get_model_file_for_farm.php"
                
                params = {}
                if farm_id:
                    params['farm_parcels_id'] = farm_id
                if device_id:
                    params['device_id'] = device_id
                
                # Try to download farm-specific model
                print(f"📥 Downloading model for {farm_id or device_id}...")
                response = requests.get(model_url, params=params, timeout=180, stream=True)
                
                if response.status_code == 200:
                    models_dir = model_cache.models_dir
                    
                    # Save as farm-specific model file
                    cache_key = model_cache.get_cache_key(farm_id, device_id)
                    farm_model_path = models_dir / f"{cache_key}.onnx"
                    
                    with open(farm_model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Load the farm-specific model
                    session, input_details, output_details = load_onnx_model(str(farm_model_path))
                    ONNX_MODEL_PATH = str(farm_model_path)
                    MODEL_VERSION = response.headers.get('X-Model-Version', 'N/A')
                    MODEL_ACCURACY = response.headers.get('X-Model-Accuracy', 'N/A')
                    
                    # Cache the model
                    model_cache.cache_model(
                        farm_id=farm_id,
                        device_id=device_id,
                        session=session,
                        input_details=input_details,
                        output_details=output_details,
                        model_path=str(farm_model_path),
                        version=MODEL_VERSION,
                        accuracy=MODEL_ACCURACY
                    )
                    
                    model_source = f"farm_{farm_id or device_id}"
                    used_farm_id = farm_id or device_id
                    print(f"✅ Loaded and cached farm-specific model: {MODEL_VERSION} (accuracy: {MODEL_ACCURACY}%)")
                elif response.status_code == 404:
                    print(f"⚠️  No farm-specific model found for {farm_id or device_id}, using default model")
                    model_source = "global_fallback"
                else:
                    print(f"⚠️  Error downloading model (HTTP {response.status_code}), using default model")
                    model_source = "global_fallback"
        except Exception as e:
            print(f"⚠️  Could not load farm-specific model ({e}), using default model")
            model_source = "global_fallback"
            # Continue with default model
    
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
        # Try to detect model type from path or output shape
        # Classification models typically have output shape [1, num_classes] or [1, num_classes, 1, 1]
        # YOLO models have output shape [1, features, detections] or [1, detections, features]
        if ONNX_MODEL_PATH and ('yolo' in ONNX_MODEL_PATH.lower()):
            input_shape = [1, 3, 640, 640]  # YOLO standard
        else:
            input_shape = [1, 3, 512, 512]  # Classification default (ResNet18 uses 224x224, but we'll resize)
    
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
    
    # Debug: Log detection results with model info
    print(f"🔍 Detection using model: {model_source} (version: {MODEL_VERSION}, accuracy: {MODEL_ACCURACY}%)")
    if used_farm_id:
        print(f"🔍 Farm/Device ID: {used_farm_id}")
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
    # Normalize pest names for comparison (case-insensitive, handle variations)
    def normalize_pest_name(name):
        """Normalize pest name for comparison"""
        if not name:
            return ""
        # Convert to lowercase, replace spaces/hyphens/underscores with single underscore
        normalized = name.lower().strip()
        normalized = normalized.replace('-', '_').replace(' ', '_')
        # Remove multiple underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        return normalized.strip('_')
    
    # Create normalized mapping of known classes
    normalized_class_names = {normalize_pest_name(name): name for name in CLASS_NAMES}
    
    # Debug: Log known classes for troubleshooting
    print(f"🔍 Verification: Known classes (normalized): {list(normalized_class_names.keys())}")
    print(f"🔍 Verification: Detected pests: {list(counts.keys())}")
    
    verified_pests = {}
    unverified_detections = []
    
    # Check each detected pest
    for pest_name, count in counts.items():
        if count > 0:
            # Normalize the detected pest name
            normalized_detected = normalize_pest_name(pest_name)
            
            # Check if normalized name matches any known class
            if normalized_detected in normalized_class_names:
                # Use the original CLASS_NAMES version for consistency
                original_name = normalized_class_names[normalized_detected]
                verified_pests[original_name] = count
                print(f"✅ Verified: '{pest_name}' -> '{original_name}' (normalized: '{normalized_detected}')")
            else:
                # Unknown pest detected
                print(f"⚠️  Unverified: '{pest_name}' (normalized: '{normalized_detected}') not in known classes")
                unverified_detections.append({
                    "pest_name": pest_name,
                    "count": count,
                    "reason": "not_in_training_data",
                    "normalized_name": normalized_detected  # For debugging
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
        # No pests detected - treat as unverified (needs review)
        verification_status = "unverified"  # No pests detected - needs manual review
        # Add a special unverified detection entry
        unverified_detections.append({
            "pest_name": "No Pest Detected",
            "count": 0,
            "reason": "no_detection",
            "detected_as": "No Pest Detected"
        })
    
    response_data = {
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
        "known_classes": CLASS_NAMES,  # List of all known pest classes
        "model_metadata": {
            "source": model_source,  # "global", "cached_farm_X", "farm_X", "global_fallback"
            "farm_id": used_farm_id,  # Farm/device ID if farm-specific model used
            "version": MODEL_VERSION or "N/A",
            "accuracy": float(MODEL_ACCURACY) if MODEL_ACCURACY else None,
            "cached": model_source.startswith("cached_")  # Whether model was from cache
        }
    }
    
    # Add model metadata if available (backward compatibility)
    if MODEL_VERSION:
        response_data["model_version"] = MODEL_VERSION
    if MODEL_ACCURACY:
        response_data["model_accuracy"] = float(MODEL_ACCURACY)
    
    return jsonify(response_data)

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

# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/cache/invalidate', methods=['POST'])
def invalidate_cache():
    """Invalidate model cache for a specific farm/device"""
    try:
        data = request.json or {}
        farm_id = data.get('farm_id') or data.get('farm_parcels_id')
        device_id = data.get('device_id')
        
        if not farm_id and not device_id:
            return jsonify({
                'success': False,
                'message': 'farm_id or device_id required'
            }), 400
        
        model_cache.invalidate_cache(farm_id=farm_id, device_id=device_id)
        
        return jsonify({
            'success': True,
            'message': f'Cache invalidated for {farm_id or device_id}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Clean up old unused models from cache"""
    try:
        max_age = request.json.get('max_age_seconds', 3600) if request.json else 3600
        model_cache.cleanup_old_models(max_age_seconds=max_age)
        
        return jsonify({
            'success': True,
            'message': 'Cache cleanup completed',
            'cached_models': len(model_cache.cache)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/cache/status', methods=['GET'])
def cache_status():
    """Get cache status"""
    try:
        cache_info = {}
        for key, cached in model_cache.cache.items():
            cache_info[key] = {
                'version': cached.get('version', 'N/A'),
                'accuracy': cached.get('accuracy', 'N/A'),
                'last_used': cached.get('last_used', 0),
                'age_seconds': time.time() - cached.get('last_used', 0)
            }
        
        return jsonify({
            'success': True,
            'total_cached': len(model_cache.cache),
            'cache': cache_info
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# STARTUP INITIALIZATION
# ============================================================================

def initialize_app():
    """Initialize app on startup"""
    print("🚀 Initializing AgriShield API...")
    
    # Clean up old cached models (older than 1 hour)
    print("🧹 Cleaning up old cached models...")
    model_cache.cleanup_old_models(max_age_seconds=3600)
    
    # Load default model
    print("📦 Loading default model...")
    try:
        model_path = find_onnx_model()
        if model_path:
            global session, input_details, output_details, ONNX_MODEL_PATH
            session, input_details, output_details = load_onnx_model(model_path)
            ONNX_MODEL_PATH = model_path
            print(f"✅ Default model loaded: {Path(model_path).name}")
        else:
            print("⚠️  No default model found")
    except Exception as e:
        print(f"⚠️  Error loading default model: {e}")
    
    print("✅ Initialization complete")

# Initialize on import (for production) or on first request
if os.getenv('FLASK_ENV') != 'development':
    initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"Starting Combined AgriShield API on port {port}...")
    print(f"  - Pest Detection: ONNX Runtime")
    print(f"  - Training Service: PyTorch")
    print(f"  - Model Caching: Enabled")
    
    # Initialize on startup
    initialize_app()
    
    app.run(host="0.0.0.0", port=port, debug=False)

