"""
Independent Training Service for Heroku
Runs training automatically without browser/Colab
"""

from flask import Flask, request, jsonify
import pymysql
import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import shutil

app = Flask(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'auth-db1322.hstgr.io'),
    'user': os.getenv('DB_USER', 'u520834156_uAShield2025'),
    'password': os.getenv('DB_PASSWORD', ':JqjB0@0zb6v'),
    'database': os.getenv('DB_NAME', 'u520834156_dbAgriShield'),
    'charset': 'utf8mb4'
}

# Training script path (will be in same directory or specified)
TRAINING_SCRIPT = os.getenv('TRAINING_SCRIPT', 'train.py')

def get_training_job(job_id):
    """Get training job from database"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM training_jobs WHERE job_id = %s", (job_id,))
        job = cursor.fetchone()
        conn.close()
        return job
    except Exception as e:
        print(f"Error getting job: {e}")
        return None

def update_job_status(job_id, status, message=None):
    """Update training job status"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute("UPDATE training_jobs SET status = %s, completed_at = NOW() WHERE job_id = %s", 
                          (status, job_id))
        elif status == 'failed':
            cursor.execute("UPDATE training_jobs SET status = %s, completed_at = NOW(), error_message = %s WHERE job_id = %s", 
                          (status, message[:500] if message else None, job_id))  # Limit error message length
        else:
            cursor.execute("UPDATE training_jobs SET status = %s WHERE job_id = %s", 
                          (status, job_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating job status: {e}")

def log_to_database(job_id, level, message):
    """Log to database"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'training_logs'")
        if cursor.fetchone():
            cursor.execute("INSERT INTO training_logs (training_job_id, log_level, message) VALUES (%s, %s, %s)",
                          (job_id, level, message[:1000]))  # Limit message length
            conn.commit()
        conn.close()
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
        
        # Always parse training_config for logging and other data (farm_id, selected_pests, etc.)
        training_config_raw = job.get('training_config', '{}')
        config = {}  # Initialize config to avoid "not defined" error
        
        # Parse training_config JSON
        if isinstance(training_config_raw, str):
            try:
                if training_config_raw and training_config_raw.strip() and training_config_raw.strip() != '{}':
                    config = json.loads(training_config_raw)
                else:
                    config = {}
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON in training_config: {training_config_raw}")
                print(f"[ERROR] JSON decode error: {e}")
                config = {}
        elif isinstance(training_config_raw, dict):
            config = training_config_raw
        else:
            config = {}
        
        # If columns don't exist or are None, fall back to training_config JSON
        if epochs is None or batch_size is None:
            print(f"[INFO] epochs/batch_size columns not found, falling back to training_config JSON")
            print(f"[DEBUG] Raw training_config from database: {training_config_raw} (type: {type(training_config_raw)})")
            
            # Extract from config if columns not available
            if epochs is None:
                epochs = config.get('epochs') if isinstance(config, dict) else None
            if batch_size is None:
                batch_size = config.get('batch_size') if isinstance(config, dict) else None
        else:
            print(f"[INFO] Using epochs/batch_size from dedicated columns (epochs={epochs}, batch_size={batch_size})")
        
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
        log_to_database(job_id, 'INFO', f'Training config parsed: epochs={epochs}, batch_size={batch_size} (from columns: {job.get("epochs") is not None})')
        
        update_job_status(job_id, 'running')
        log_to_database(job_id, 'INFO', 'Training started on Heroku service')
        
        # Check if training script exists
        script_path = os.path.join(os.path.dirname(__file__), TRAINING_SCRIPT)
        if not os.path.exists(script_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.getcwd(), TRAINING_SCRIPT),
                os.path.join(os.getcwd(), 'ml_deployment', 'scripts', 'admin_training_script.py'),
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

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'AgriShield Training Service',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'train': '/train (POST)',
            'train_classification': '/train/classification (POST)',
            'status': '/status/<job_id> (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = pymysql.connect(**DB_CONFIG)
        conn.close()
        return jsonify({'status': 'ok', 'service': 'training-service', 'database': 'connected'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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

def run_classification_training(dataset_path, save_dir, epochs, batch_size, img_size, model_name, num_classes):
    """Run ResNet18 classification training via subprocess"""
    try:
        # Create training script path
        script_dir = Path(__file__).resolve().parent
        train_script = script_dir / 'train_classification.py'
        
        # If dedicated script doesn't exist, use main train.py with modified approach
        if not train_script.exists():
            train_script = script_dir / 'train.py'
        
        # Build command
        cmd = [
            'python', str(train_script),
            '--dataset_path', str(dataset_path),
            '--save_dir', str(save_dir),
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--img_size', str(img_size),
            '--model', model_name,
            '--num_classes', str(num_classes)
        ]
        
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            cwd=str(script_dir)
        )
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, '', 'Training timeout (exceeded 2 hours)'
    except Exception as e:
        return False, '', str(e)

@app.route('/train/classification', methods=['POST'])
def train_classification():
    """
    Train ResNet18 classification model
    
    Expected JSON:
    {
        "dataset_path": "/path/to/dataset",  # Required
        "epochs": 50,                        # Optional, default 50
        "batch": 16,                         # Optional, default 16
        "img_size": 224,                     # Optional, default 224
        "model": "resnet18",                 # Optional, default "resnet18"
        "num_classes": 5                      # Required
    }
    """
    try:
        data = request.json or {}
        
        # Validate required parameters
        dataset_path = data.get('dataset_path')
        if not dataset_path:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: dataset_path'
            }), 400
        
        num_classes = data.get('num_classes')
        if num_classes is None:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: num_classes'
            }), 400
        
        # Get optional parameters with defaults
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch', 16)
        img_size = data.get('img_size', 224)
        model_name = data.get('model', 'resnet18')
        
        # Validate dataset path exists
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            return jsonify({
                'success': False,
                'error': f'Dataset path does not exist: {dataset_path}'
            }), 400
        
        if not dataset_path_obj.is_dir():
            return jsonify({
                'success': False,
                'error': f'Dataset path is not a directory: {dataset_path}'
            }), 400
        
        # Handle different dataset structures
        # Check for 100.v1i.folder structure first
        roboflow_dir = dataset_path_obj / '100.v1i.folder'
        if roboflow_dir.exists() and roboflow_dir.is_dir():
            dataset_path_obj = roboflow_dir
        
        # Check for classification structure
        classification_dir = dataset_path_obj / 'classification'
        if classification_dir.exists() and classification_dir.is_dir():
            dataset_path_obj = classification_dir
        
        # Verify required folders exist
        train_dir = dataset_path_obj / 'train'
        valid_dir = dataset_path_obj / 'valid'
        test_dir = dataset_path_obj / 'test'
        
        # Check for valid/val naming variation
        if not valid_dir.exists():
            val_dir = dataset_path_obj / 'val'
            if val_dir.exists():
                valid_dir = val_dir
        
        missing_folders = []
        if not train_dir.exists():
            missing_folders.append('train/')
        if not valid_dir.exists():
            missing_folders.append('valid/ or val/')
        
        if missing_folders:
            return jsonify({
                'success': False,
                'error': f'Missing required folders: {", ".join(missing_folders)}',
                'dataset_path': str(dataset_path),
                'resolved_path': str(dataset_path_obj),
                'checked': {
                    'train': train_dir.exists(),
                    'valid': valid_dir.exists(),
                    'test': test_dir.exists()
                },
                'hint': 'Dataset should have train/ and valid/ (or val/) folders. If using 100.v1i.folder structure, ensure it contains train/ and valid/ subfolders.'
            }), 400
        
        # Create timestamped save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_dir = Path(__file__).resolve().parent
        runs_dir = script_dir / 'runs' / 'classification'
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        save_dir = runs_dir / f'train_{timestamp}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if epochs < 1:
            return jsonify({
                'success': False,
                'error': 'epochs must be >= 1'
            }), 400
        
        if batch_size < 1:
            return jsonify({
                'success': False,
                'error': 'batch must be >= 1'
            }), 400
        
        if num_classes < 1:
            return jsonify({
                'success': False,
                'error': 'num_classes must be >= 1'
            }), 400
        
        if model_name not in ['resnet18', 'resnet34', 'resnet50']:
            return jsonify({
                'success': False,
                'error': f'Unsupported model: {model_name}. Supported: resnet18, resnet34, resnet50'
            }), 400
        
        # Start training in background thread
        def training_worker():
            try:
                # Pass the resolved dataset path (which may be inside 100.v1i.folder or classification)
                success, stdout, stderr = run_classification_training(
                    dataset_path_obj,  # This is already resolved to the correct path
                    save_dir,
                    epochs,
                    batch_size,
                    img_size,
                    model_name,
                    num_classes
                )
                
                # Save training output
                output_file = save_dir / 'training_output.txt'
                with open(output_file, 'w') as f:
                    f.write("=== STDOUT ===\n")
                    f.write(stdout)
                    f.write("\n\n=== STDERR ===\n")
                    f.write(stderr)
                
                if not success:
                    error_file = save_dir / 'error.txt'
                    with open(error_file, 'w') as f:
                        f.write(stderr or stdout or 'Unknown error')
                
            except Exception as e:
                error_file = save_dir / 'error.txt'
                with open(error_file, 'w') as f:
                    f.write(f'Training error: {str(e)}')
        
        thread = threading.Thread(target=training_worker)
        thread.daemon = True
        thread.start()
        
        # Return immediate response
        return jsonify({
            'success': True,
            'message': 'Classification training started',
            'training_status': 'running',
            'save_directory': str(save_dir),
            'parameters': {
                'dataset_path': str(dataset_path),
                'resolved_path': str(dataset_path_obj),
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'model': model_name,
                'num_classes': num_classes
            },
            'dataset_structure': {
                'train': train_dir.exists(),
                'valid': valid_dir.exists(),
                'test': test_dir.exists()
            },
            'note': 'If dataset_path contains 100.v1i.folder or classification, it will be automatically resolved'
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            'success': False,
            'error': 'Failed to start training',
            'message': error_msg
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

