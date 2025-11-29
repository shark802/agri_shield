"""
Independent Training Service for Railway/Render
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

# Database configuration from environment
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'auth-db1322.hstgr.io'),
    'user': os.getenv('DB_USER', 'u520834156_uAShield2025'),
    'password': os.getenv('DB_PASSWORD', ':JqjB0@0zb6v'),
    'database': os.getenv('DB_NAME', 'u520834156_dbAgriShield'),
    'charset': 'utf8mb4'
}

# Training script path
TRAINING_SCRIPT = os.getenv('TRAINING_SCRIPT', '/app/train.py')

def get_training_job(job_id):
    """Get training job from database"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM training_jobs WHERE job_id = %s", (job_id,))
    job = cursor.fetchone()
    conn.close()
    return job

def update_job_status(job_id, status, message=None):
    """Update training job status"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    if status == 'completed':
        cursor.execute("UPDATE training_jobs SET status = %s, completed_at = NOW() WHERE job_id = %s", 
                      (status, job_id))
    elif status == 'failed':
        cursor.execute("UPDATE training_jobs SET status = %s, completed_at = NOW(), error_message = %s WHERE job_id = %s", 
                      (status, message, job_id))
    else:
        cursor.execute("UPDATE training_jobs SET status = %s WHERE job_id = %s", 
                      (status, job_id))
    
    conn.commit()
    conn.close()

def log_to_database(job_id, level, message):
    """Log to database"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'training_logs'")
        if cursor.fetchone():
            cursor.execute("INSERT INTO training_logs (training_job_id, log_level, message) VALUES (%s, %s, %s)",
                          (job_id, level, message))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Log error: {e}")

def run_training(job_id):
    """Run training in background thread"""
    try:
        job = get_training_job(job_id)
        if not job:
            return
        
        config = json.loads(job['training_config'])
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 8)
        
        update_job_status(job_id, 'running')
        log_to_database(job_id, 'INFO', 'Training started on cloud service')
        
        # Run training script
        cmd = [
            'python', TRAINING_SCRIPT,
            '--job_id', str(job_id),
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            update_job_status(job_id, 'completed')
            log_to_database(job_id, 'INFO', 'Training completed successfully')
        else:
            error_msg = result.stderr[:500]  # Limit error message length
            update_job_status(job_id, 'failed', error_msg)
            log_to_database(job_id, 'ERROR', f'Training failed: {error_msg}')
            
    except Exception as e:
        error_msg = str(e)[:500]
        update_job_status(job_id, 'failed', error_msg)
        log_to_database(job_id, 'ERROR', f'Training error: {error_msg}')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'training-service'})

@app.route('/train', methods=['POST'])
def start_training():
    """Start training job"""
    try:
        data = request.json
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'success': False, 'message': 'job_id required'}), 400
        
        # Check if job exists and is pending
        job = get_training_job(job_id)
        if not job:
            return jsonify({'success': False, 'message': 'Job not found'}), 404
        
        if job['status'] != 'pending':
            return jsonify({'success': False, 'message': f'Job status is {job["status"]}, expected pending'}), 400
        
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

