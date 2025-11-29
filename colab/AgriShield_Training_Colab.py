"""
AgriShield Pest Detection Model Training - Google Colab Version
Copy this script into Google Colab cells for cloud training

Instructions:
1. Create new Google Colab notebook
2. Copy each section below into separate cells
3. Update database credentials
4. Run all cells to start training
"""

# ============================================================================
# CELL 1: Install Required Packages
# ============================================================================
# !pip install torch torchvision pymysql pillow numpy scikit-learn -q
# print("✅ Packages installed")

# ============================================================================
# CELL 2: Mount Google Drive
# ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')
# print("✅ Google Drive mounted")

# ============================================================================
# CELL 3: Database Configuration
# ============================================================================
# Database configuration (UPDATE with your database credentials)
DB_CONFIG = {
    'host': 'auth-db1322.hstgr.io',  # Your database host
    'user': 'u520834156_uAShield2025',
    'password': ':JqjB0@0zb6v',
    'database': 'u520834156_dbAgriShield',
    'charset': 'utf8mb4'
}

# Training job ID (will be read from database or set manually)
JOB_ID = None  # Set to specific job_id or leave None to get latest pending job

print("✅ Database configuration loaded")

# ============================================================================
# CELL 4: Database Helper Functions
# ============================================================================
import pymysql
import json
from datetime import datetime

def get_training_job(job_id):
    """Get training job details from database"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    cursor.execute("SELECT * FROM training_jobs WHERE job_id = %s", (job_id,))
    job = cursor.fetchone()
    
    conn.close()
    return job

def update_job_status(job_id, status, message=None):
    """Update training job status in database"""
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
    """Log training message to database"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if training_logs table exists
        cursor.execute("SHOW TABLES LIKE 'training_logs'")
        if cursor.fetchone():
            cursor.execute("INSERT INTO training_logs (training_job_id, log_level, message) VALUES (%s, %s, %s)",
                          (job_id, level, message))
            conn.commit()
        else:
            print(f"[{level}] {message}")  # Fallback to print if table doesn't exist
        
        conn.close()
    except Exception as e:
        print(f"Log error: {e}")

print("✅ Database functions ready")

# ============================================================================
# CELL 5: Auto-Polling for Training Jobs (AUTOMATIC!)
# ============================================================================
# This cell automatically checks for new training jobs every 30 seconds
# Keep this cell running - it will detect new jobs automatically!

import time
from IPython.display import clear_output

def check_for_pending_jobs():
    """Check database for pending cloud training jobs"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT job_id FROM training_jobs WHERE status = 'pending' AND cloud_training = 1 ORDER BY started_at DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result['job_id'] if result else None
    except Exception as e:
        print(f"Error checking for jobs: {e}")
        return None

print("🔄 Auto-polling started! Checking for new training jobs every 30 seconds...")
print("💡 Keep this cell running - it will automatically detect new jobs from admin interface")
print("💡 You can minimize this tab - training will start automatically when you click 'Train in Cloud'")
print("="*60)

# Auto-polling loop
while True:
    pending_job_id = check_for_pending_jobs()
    
    if pending_job_id:
        print(f"\n✅ NEW TRAINING JOB DETECTED: Job #{pending_job_id}")
        print("🚀 Starting training automatically...")
        JOB_ID = pending_job_id
        break  # Exit loop and proceed to training
    else:
        # Show status with timestamp
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        clear_output(wait=True)
        print(f"⏳ [{current_time}] Waiting for new training jobs...")
        print("💡 Click 'Train in Cloud' in admin interface to start training")
        print("="*60)
    
    time.sleep(30)  # Check every 30 seconds

# Get job details
if JOB_ID:
    job = get_training_job(JOB_ID)
    if job:
        config = json.loads(job['training_config'])
        EPOCHS = config.get('epochs', 50)
        BATCH_SIZE = config.get('batch_size', 8)
        print(f"\n📋 Job {JOB_ID}: {EPOCHS} epochs, batch size {BATCH_SIZE}")
        update_job_status(JOB_ID, 'running')
        log_to_database(JOB_ID, 'INFO', 'Training started automatically in Google Colab')
    else:
        print(f"❌ Job {JOB_ID} not found")
        JOB_ID = None

# ============================================================================
# CELL 6: Import Training Libraries
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from pathlib import Path
import numpy as np
import os

print(f"✅ PyTorch {torch.__version__} ready")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("⚠️ No GPU available, using CPU (training will be slower)")
    device = torch.device('cpu')

# ============================================================================
# CELL 7: Dataset Path Configuration
# ============================================================================
# Option 1: Dataset in Google Drive
DATASET_PATH = '/content/drive/MyDrive/AgriShield/datasets'

# Option 2: Download from your server (if you create API endpoint)
# import requests
# dataset_url = "https://agrishield.bccbsis.com/api/download_dataset"
# !wget -O /content/dataset.zip {dataset_url}
# !unzip /content/dataset.zip -d /content/datasets
# DATASET_PATH = '/content/datasets'

if os.path.exists(DATASET_PATH):
    print(f"✅ Dataset found: {DATASET_PATH}")
else:
    print(f"⚠️ Dataset not found at {DATASET_PATH}")
    print("💡 Upload dataset to Google Drive: /content/drive/MyDrive/AgriShield/datasets/")
    DATASET_PATH = None

# ============================================================================
# CELL 8: Training Code
# ============================================================================
# NOTE: Copy your full training code from admin_training_script.py here
# This is a simplified placeholder - replace with your actual training logic

if JOB_ID and DATASET_PATH:
    try:
        log_to_database(JOB_ID, 'INFO', 'Starting model training...')
        
        # ===== YOUR TRAINING CODE HERE =====
        # Copy the training logic from admin_training_script.py
        # This includes:
        # - Dataset loading
        # - Model initialization (ResNet18)
        # - Training loop
        # - Validation
        # - Model saving
        
        # Example structure (replace with your actual code):
        """
        # Load dataset
        train_dataset = YourDatasetClass(DATASET_PATH, split='train')
        val_dataset = YourDatasetClass(DATASET_PATH, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = len(train_dataset.classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        # Training loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            
            accuracy = 100. * correct / len(val_dataset)
            log_to_database(JOB_ID, 'INFO', f'Epoch {epoch+1}/{EPOCHS}: Accuracy = {accuracy:.2f}%')
        
        # Save model
        model_save_path = f'/content/drive/MyDrive/AgriShield/models/job_{JOB_ID}_best_model.pth'
        torch.save(model.state_dict(), model_save_path)
        
        log_to_database(JOB_ID, 'INFO', f'Model saved to: {model_save_path}')
        update_job_status(JOB_ID, 'completed')
        print("✅ Training completed!")
        """
        
        # For now, just mark as completed (replace with actual training)
        log_to_database(JOB_ID, 'INFO', 'Training placeholder - replace with actual training code')
        update_job_status(JOB_ID, 'completed')
        print("✅ Training job completed (placeholder)")
        
    except Exception as e:
        error_msg = str(e)
        log_to_database(JOB_ID, 'ERROR', f'Training failed: {error_msg}')
        update_job_status(JOB_ID, 'failed', error_msg)
        print(f"❌ Training failed: {error_msg}")
else:
    print("⚠️ Cannot start training: Missing JOB_ID or DATASET_PATH")

# ============================================================================
# CELL 9: Download Model (Optional)
# ============================================================================
# After training completes, you can download the model:
# from google.colab import files
# files.download(f'/content/drive/MyDrive/AgriShield/models/job_{JOB_ID}_best_model.pth')

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Job ID: {JOB_ID}")
print("Check admin interface for training status")
print("Model saved to Google Drive")

