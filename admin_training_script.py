#!/usr/bin/env python3
"""
Enhanced Training Script for Admin Training Module
Integrates with the web interface for real-time monitoring
"""

# Force unbuffered output from the very start
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

print("=" * 60, flush=True)
print("SCRIPT STARTING", flush=True)
print("=" * 60, flush=True)
print(f"Python: {sys.executable}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
sys.stdout.flush()

import json
import time
import logging
import argparse
import pymysql
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import re

print("[OK] All imports successful", flush=True)
sys.stdout.flush()
# Optional imports - make training work even if these aren't installed
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available - metrics will be limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available - plots will be skipped")

# ============================================================================
# LOAD FROM config.php (like Flask app does)
# ============================================================================

def load_config_from_php():
    """Load configuration from config.php file (same as Flask config.py)"""
    config = {}
    # config.php is in parent directory (Proto1/)
    config_php_path = Path(__file__).resolve().parent.parent / 'config.php'
    
    if config_php_path.exists():
        try:
            with open(config_php_path, 'r') as f:
                content = f.read()
            
            # Extract DB_HOST
            match = re.search(r"define\s*\(\s*['\"]DB_HOST['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_host'] = match.group(1)
            
            # Extract DB_USER
            match = re.search(r"define\s*\(\s*['\"]DB_USER['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_user'] = match.group(1)
            
            # Extract DB_PASS
            match = re.search(r"define\s*\(\s*['\"]DB_PASS['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_password'] = match.group(1)
            
            # Extract DB_NAME
            match = re.search(r"define\s*\(\s*['\"]DB_NAME['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_name'] = match.group(1)
            
            print(f"[OK] Loaded config from config.php: {config.get('db_host')} / {config.get('db_name')}", flush=True)
        except Exception as e:
            print(f"Warning: Could not read config.php: {e}", flush=True)
    else:
        print(f"Warning: config.php not found at {config_php_path}", flush=True)
    
    return config

# Load from config.php
php_config = load_config_from_php()

# Database configuration
# Priority: Environment variables > config.php > defaults
DB_CONFIG = {
    'host': os.getenv('DB_HOST', php_config.get('db_host', 'localhost')),
    'user': os.getenv('DB_USER', php_config.get('db_user', 'root')),
    'password': os.getenv('DB_PASSWORD', php_config.get('db_password', '')),
    'database': os.getenv('DB_NAME', php_config.get('db_name', 'asdb')),
    'charset': os.getenv('DB_CHARSET', 'utf8mb4')
}

print(f"[OK] Database config: {DB_CONFIG['host']} / {DB_CONFIG['database']}", flush=True)

class AdminTrainingLogger:
    """Custom logger that writes to database"""
    
    def __init__(self, job_id, db_config):
        self.job_id = job_id
        self.db_config = db_config
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration - XAMPP compatible"""
        # Logs go to parent directory (Proto1/training_logs/)
        script_dir = Path(__file__).resolve().parent.parent
        log_dir = script_dir / "training_logs"
        log_dir.mkdir(exist_ok=True)
        # XAMPP: Ensure directory is writable
        try:
            import os
            os.chmod(str(log_dir), 0o777)
        except:
            pass
        
        # Force unbuffered output
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(line_buffering=True)
        
        log_file_path = log_dir / f"job_{self.job_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # Immediately write to log file to confirm it works
        self.logger.info(f"Log file created: {log_file_path.absolute()}")
        print(f"Logging to: {log_file_path.absolute()}", flush=True)
        sys.stdout.flush()
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
        self.log_to_db('INFO', message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
        self.log_to_db('WARNING', message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
        self.log_to_db('ERROR', message)
    
    def log_to_db(self, level, message):
        """Log message to database"""
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_logs (training_job_id, log_level, message) VALUES (%s, %s, %s)",
                (self.job_id, level, message)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to log to database: {e}")

class EnhancedPestDataset(Dataset):
    """Enhanced dataset with better error handling and statistics"""
    
    def __init__(self, data_dir, transform=None, logger=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.logger = logger
        self.samples = []
        self.class_counts = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset with statistics"""
        if not self.data_dir.exists():
            if self.logger:
                self.logger.error(f"Dataset directory not found: {self.data_dir}")
            return
        
        # Get pest classes
        self.classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        if self.logger:
            self.logger.info(f"Found classes: {self.classes}")
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_images = []
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
                    class_images.append(img_path)
            
            self.class_counts[class_name] = len(class_images)
            
            if self.logger:
                self.logger.info(f"{class_name}: {len(class_images)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_statistics(self):
        """Get dataset statistics"""
        total_images = len(self.samples)
        return {
            'total_images': total_images,
            'class_counts': self.class_counts,
            'classes': self.classes
        }

class ModelTrainer:
    """Enhanced model trainer with database integration"""
    
    def __init__(self, job_id, config, logger):
        self.job_id = job_id
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_model(self, num_classes):
        """Create ResNet18 model for classification"""
        self.logger.info(f"Creating ResNet18 model with {num_classes} classes")
        
        # Load pre-trained ResNet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final layer for our number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model
    
    def get_data_transforms(self):
        """Get data transforms for training and validation"""
        
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch - YOLOv8 style output"""
        import sys
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # YOLOv8-style progress output (like CMD training)
            batch_progress = (batch_idx + 1) / len(dataloader) * 100
            current_acc = 100. * correct / total if total > 0 else 0
            
            # Print progress every batch (like YOLOv8)
            progress_bar_length = 30
            filled = int(progress_bar_length * batch_progress / 100)
            bar = '█' * filled + '░' * (progress_bar_length - filled)
            
            # Format like YOLOv8: epoch/batch  loss  accuracy  progress_bar
            progress_line = f"  {batch_idx+1}/{len(dataloader)}  {loss.item():.4f}  {current_acc:.1f}%  {bar} {batch_progress:.0f}%"
            print(progress_line, end='\r', flush=True)
            
            # Also log every 10 batches
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                self.logger.info(f'Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
                sys.stdout.flush()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, dataloader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_metrics_to_db(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Save training metrics to database"""
        try:
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_metrics (training_job_id, epoch, accuracy, loss, val_accuracy, val_loss) VALUES (%s, %s, %s, %s, %s, %s)",
                (self.job_id, epoch, train_acc/100, train_loss, val_acc/100, val_loss)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save metrics to database: {e}")
    
    def train(self, train_dataset, val_dataset):
        """Main training function"""
        self.logger.info("Starting training process")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        
        # Create model
        num_classes = len(train_dataset.classes)
        model = self.create_model(num_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop - YOLOv8 style output
        import sys
        print("\n" + "="*60, flush=True)
        print("TRAINING STARTED", flush=True)
        print("="*60, flush=True)
        print(f"Epochs: {self.config['epochs']}", flush=True)
        print(f"Batch Size: {self.config['batch_size']}", flush=True)
        print(f"Learning Rate: {self.config['learning_rate']}", flush=True)
        print(f"Train Batches: {len(train_loader)}", flush=True)
        print(f"Val Batches: {len(val_loader)}", flush=True)
        print("="*60 + "\n", flush=True)
        sys.stdout.flush()
        
        for epoch in range(self.config['epochs']):
            # YOLOv8-style epoch header
            print(f"\n{'='*60}", flush=True)
            print(f"Epoch {epoch+1}/{self.config['epochs']}", flush=True)
            print(f"{'='*60}", flush=True)
            sys.stdout.flush()
            
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Train with progress output
            print(f"\nTraining:", flush=True)
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            print()  # New line after progress bar
            
            # Validate
            print(f"\nValidating:", flush=True)
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Save metrics
            self.save_metrics_to_db(epoch+1, train_loss, train_acc, val_loss, val_acc)
            
            # Store in history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # YOLOv8-style epoch summary
            print(f"\nEpoch {epoch+1} Summary:", flush=True)
            print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%", flush=True)
            print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%", flush=True)
            sys.stdout.flush()
            
            # Log epoch results
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model(model, val_acc, train_dataset.classes)
                print(f"  [OK] New best model saved! (Accuracy: {val_acc:.2f}%)", flush=True)
                sys.stdout.flush()
        
        return model
    
    def save_model(self, model, accuracy, classes):
        """Save model to database and file system - Auto-activates new model"""
        try:
            # Create model directory (in parent directory, same as root)
            script_dir = Path(__file__).resolve().parent.parent
            model_dir = script_dir / "models" / f"job_{self.job_id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_path = model_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Get model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Save to database
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            # Generate version number
            cursor.execute("SELECT MAX(CAST(SUBSTRING(version, 2) AS UNSIGNED)) FROM model_versions")
            result = cursor.fetchone()
            next_version = (result[0] or 0) + 1
            
            # Deactivate all previous models
            cursor.execute("UPDATE model_versions SET is_active = 0, is_current = 0")
            
            # Insert new model as active
            cursor.execute(
                "INSERT INTO model_versions (version, model_path, accuracy, training_job_id, model_size_mb, is_active, is_current, deployed_at) VALUES (%s, %s, %s, %s, %s, 1, 1, NOW())",
                (f"v{next_version}", str(model_path), accuracy/100, self.job_id, model_size_mb)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"[OK] Model v{next_version} saved and automatically activated with accuracy: {accuracy:.2f}%")
            self.logger.info(f"[INFO] All previous models have been deactivated")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

def update_job_status(job_id, status, error_message=None):
    """Update training job status in database"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute(
                "UPDATE training_jobs SET status = %s, completed_at = NOW() WHERE job_id = %s",
                (status, job_id)
            )
        elif status == 'failed':
            cursor.execute(
                "UPDATE training_jobs SET status = %s, completed_at = NOW(), error_message = %s WHERE job_id = %s",
                (status, error_message, job_id)
            )
        else:
            cursor.execute(
                "UPDATE training_jobs SET status = %s WHERE job_id = %s",
                (status, job_id)
            )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to update job status: {e}")

def create_combined_dataset(logger):
    """Create combined dataset from original and collected data"""
    import sys
    print("Creating combined dataset...", flush=True)
    logger.info("Creating combined dataset...")
    
    # Directories - use parent directory (Proto1/)
    script_dir = Path(__file__).resolve().parent.parent
    original_train_dir = script_dir / "ml_training" / "datasets" / "processed" / "train"
    original_val_dir = script_dir / "ml_training" / "datasets" / "processed" / "val"
    collected_data_dir = script_dir / "ml_training" / "datasets" / "auto_collected"
    combined_dir = script_dir / "ml_training" / "datasets" / "combined"
    
    print(f"Checking directories...", flush=True)
    print(f"  Train dir: {original_train_dir} (exists: {original_train_dir.exists()})", flush=True)
    print(f"  Val dir: {original_val_dir} (exists: {original_val_dir.exists()})", flush=True)
    sys.stdout.flush()
    
    # If processed directories don't exist, use them directly
    if not original_train_dir.exists():
        print(f"WARNING: {original_train_dir} not found, checking alternatives...", flush=True)
        # Try alternative paths
        alt_paths = [
            script_dir / "datasets" / "processed" / "train",
            script_dir / "training_data" / "train",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                original_train_dir = alt_path
                original_val_dir = script_dir / alt_path.parent / "val"
                print(f"Using alternative: {original_train_dir}", flush=True)
                break
        else:
            raise FileNotFoundError(f"Training data directory not found. Checked: {original_train_dir}")
    
    # SIMPLIFIED: Use existing directories directly (no copying = much faster!)
    if original_train_dir.exists():
        if original_val_dir.exists():
            print(f"[OK] Using existing directories directly", flush=True)
            print(f"  Train: {original_train_dir}", flush=True)
            print(f"  Val: {original_val_dir}", flush=True)
            sys.stdout.flush()
            return original_train_dir, original_val_dir
        else:
            print(f"[WARN] Val dir missing, using train for both", flush=True)
            return original_train_dir, original_train_dir
    
    # Only create combined dataset if original doesn't exist
    print("Creating combined dataset structure...", flush=True)
    combined_train_dir = combined_dir / "train"
    combined_val_dir = combined_dir / "val"
    
    for split_dir in [combined_train_dir, combined_val_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in ['leptocorisa_oratorius', 'nephotettix_virescens', 'nilaparvata_lugens', 'scotinophara_coarctata']:
            (split_dir / class_name).mkdir(exist_ok=True)
    
    # Copy original training data
    logger.info("Copying original training data...")
    for class_name in ['leptocorisa_oratorius', 'nephotettix_virescens', 'nilaparvata_lugens', 'scotinophara_coarctata']:
        # Copy original train data
        original_train_class_dir = original_train_dir / class_name
        if original_train_class_dir.exists():
            for img_file in original_train_class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = combined_train_dir / class_name / f"original_{img_file.name}"
                    shutil.copy2(img_file, dest)
        
        # Copy original val data
        original_val_class_dir = original_val_dir / class_name
        if original_val_class_dir.exists():
            for img_file in original_val_class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = combined_val_dir / class_name / f"original_{img_file.name}"
                    shutil.copy2(img_file, dest)
    
    # Copy collected data (80% train, 20% val)
    logger.info("Adding auto-collected data...")
    import random
    random.seed(42)
    
    for class_name in ['leptocorisa_oratorius', 'nephotettix_virescens', 'nilaparvata_lugens', 'scotinophara_coarctata']:
        collected_class_dir = collected_data_dir / class_name
        if collected_class_dir.exists():
            images = [f for f in collected_class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if len(images) > 0:
                # Shuffle and split collected data
                random.shuffle(images)
                split_point = int(len(images) * 0.8)
                
                train_images = images[:split_point]
                val_images = images[split_point:]
                
                # Copy to train
                for img in train_images:
                    dest = combined_train_dir / class_name / f"collected_{img.name}"
                    shutil.copy2(img, dest)
                
                # Copy to val
                for img in val_images:
                    dest = combined_val_dir / class_name / f"collected_{img.name}"
                    shutil.copy2(img, dest)
                
                logger.info(f"{class_name}: +{len(images)} collected images ({len(train_images)} train, {len(val_images)} val)")
    
    return combined_train_dir, combined_val_dir

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Admin Training Script')
    parser.add_argument('--job_id', type=int, required=True, help='Training job ID')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = AdminTrainingLogger(args.job_id, DB_CONFIG)
    
    try:
        # Force output immediately
        import sys
        print("=" * 50, flush=True)
        print("PEST DETECTION TRAINING", flush=True)
        print("=" * 50, flush=True)
        print(f"Job ID: {args.job_id}", flush=True)
        print(f"Epochs: {args.epochs}", flush=True)
        print(f"Batch Size: {args.batch_size}", flush=True)
        print(f"Learning Rate: {args.learning_rate}", flush=True)
        print("=" * 50, flush=True)
        sys.stdout.flush()
        
        logger.info(f"Starting training job {args.job_id}")
        logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        
        # Update job status to running
        update_job_status(args.job_id, 'running')
        
        # Create combined dataset - with error handling
        try:
            print("Creating dataset...", flush=True)
            sys.stdout.flush()
            train_dir, val_dir = create_combined_dataset(logger)
            print(f"[OK] Dataset created: Train={train_dir}, Val={val_dir}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            error_msg = f"Failed to create dataset: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"ERROR: {error_msg}", flush=True)
            print(f"Traceback: {traceback_str}", flush=True)
            logger.error(error_msg)
            logger.error(traceback_str)
            update_job_status(args.job_id, 'failed', error_msg)
            sys.exit(1)
        
        # Get data transforms
        print("Creating model trainer...", flush=True)
        sys.stdout.flush()
        trainer = ModelTrainer(args.job_id, {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }, logger)
        
        print("Getting data transforms...", flush=True)
        sys.stdout.flush()
        train_transforms, val_transforms = trainer.get_data_transforms()
        
        # Create datasets
        print("Loading datasets...", flush=True)
        sys.stdout.flush()
        logger.info("Loading datasets...")
        train_dataset = EnhancedPestDataset(train_dir, transform=train_transforms, logger=logger)
        val_dataset = EnhancedPestDataset(val_dir, transform=val_transforms, logger=logger)
        print(f"[OK] Datasets loaded: Train={len(train_dataset)} samples, Val={len(val_dataset)} samples", flush=True)
        sys.stdout.flush()
        
        # Log dataset statistics
        train_stats = train_dataset.get_statistics()
        val_stats = val_dataset.get_statistics()
        
        logger.info(f"Training dataset: {train_stats}")
        logger.info(f"Validation dataset: {val_stats}")
        
        # Start training
        model = trainer.train(train_dataset, val_dataset)
        
        # Update job status to completed
        update_job_status(args.job_id, 'completed')
        logger.info(f"Training completed successfully! Best accuracy: {trainer.best_accuracy:.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        update_job_status(args.job_id, 'failed', str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
