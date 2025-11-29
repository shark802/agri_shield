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

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

