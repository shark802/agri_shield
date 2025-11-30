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
# pymysql is optional - we use PHP API gateway instead
try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    print("[INFO] pymysql not available - using PHP API gateway for database access", flush=True)
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
import yaml

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
# Priority: Environment variables > local defaults > config.php
# Use local database by default for training (online DB may not be accessible)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),  # Default to local root user
    'password': os.getenv('DB_PASSWORD', ''),  # Default to empty (local XAMPP)
    'database': os.getenv('DB_NAME', 'asdb'),  # Default to local database
    'charset': os.getenv('DB_CHARSET', 'utf8mb4')
}

# Database config - only needed if pymysql is available and we're running locally
# On Heroku, we use PHP API gateway instead
if PYMYSQL_AVAILABLE and not os.getenv('DB_HOST') and not os.getenv('DB_USER'):
    # Check if we can connect to local database first (local development only)
    try:
        test_conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='asdb'
        )
        test_conn.close()
        print("[OK] Using local database (localhost/root/asdb)", flush=True)
        DB_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'asdb',
            'charset': 'utf8mb4'
        }
    except Exception:
        # If local fails, try config.php credentials
        print("[WARNING] Local database not accessible, trying config.php credentials...", flush=True)
        DB_CONFIG = {
            'host': php_config.get('db_host', 'localhost'),
            'user': php_config.get('db_user', 'root'),
            'password': php_config.get('db_password', ''),
            'database': php_config.get('db_name', 'asdb'),
            'charset': 'utf8mb4'
        }
        print(f"[INFO] Using database from config.php: {DB_CONFIG['host']} / {DB_CONFIG['database']}", flush=True)
else:
    # On Heroku or when pymysql is not available, use PHP API gateway
    DB_CONFIG = None
    print("[INFO] Using PHP API gateway for database access (no direct MySQL connection)", flush=True)

class AdminTrainingLogger:
    """Custom logger that writes to database"""
    
    def __init__(self, job_id, db_config):
        self.job_id = job_id
        self.db_config = db_config
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration - XAMPP compatible with Windows permission handling"""
        import sys
        import os
        
        # Logs go to parent directory (Proto1/training_logs/)
        script_dir = Path(__file__).resolve().parent
        log_dir = script_dir / "training_logs"
        
        # Create directory with proper error handling
        try:
            log_dir.mkdir(exist_ok=True)
        except PermissionError:
            # Use ASCII-safe message for Windows
            print(f"[WARNING] Cannot create training_logs directory. Using fallback location.", flush=True)
            # Fallback to temp directory or current directory
            log_dir = Path.cwd() / "training_logs"
            try:
                log_dir.mkdir(exist_ok=True)
            except Exception as e:
                # Use ASCII-safe message for Windows
                print(f"[WARNING] Cannot create fallback log directory: {e}", flush=True)
                # Last resort: use current directory
                log_dir = Path.cwd()
        
        # Windows: Try to set permissions (may not work on Windows)
        try:
            if os.name != 'nt':  # Not Windows
                os.chmod(str(log_dir), 0o777)
        except Exception:
            pass  # Ignore permission errors on Windows
        
        # Force unbuffered output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(line_buffering=True)
        
        log_file_path = log_dir / f"job_{self.job_id}.log"
        
        # Try to create log file with error handling
        handlers = [logging.StreamHandler(sys.stdout)]  # Always include console
        
        try:
            # Test if we can write to the log file by actually trying to open it
            test_handle = None
            try:
                test_handle = open(log_file_path, 'a', encoding='utf-8')
                test_handle.write("")  # Try to write
                test_handle.close()
                
                # If successful, add file handler
                handlers.append(logging.FileHandler(log_file_path, mode='a', encoding='utf-8'))
                # Use ASCII-safe message for Windows
                print(f"[OK] Logging to file: {log_file_path.absolute()}", flush=True)
            except (PermissionError, OSError) as e:
                if test_handle:
                    test_handle.close()
                raise  # Re-raise to outer except
        except (PermissionError, OSError) as e:
            # Use ASCII-safe messages for Windows compatibility
            print(f"[WARNING] Cannot write to log file {log_file_path}: {e}", flush=True)
            print(f"[WARNING] Logging to console only. Check directory permissions.", flush=True)
            # Continue with console-only logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # Immediately write to log file to confirm it works (if file handler exists)
        try:
            if len(handlers) > 1:  # File handler was added
                self.logger.info(f"Log file created: {log_file_path.absolute()}")
            else:
                self.logger.info("Logging to console only (file logging unavailable)")
        except Exception:
            # If even logging fails, just continue silently
            pass
        sys.stdout.flush()
    
    def info(self, message):
        """Log info message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.info(safe_message)
        self.log_to_db('INFO', safe_message)
    
    def warning(self, message):
        """Log warning message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.warning(safe_message)
        self.log_to_db('WARNING', safe_message)
    
    def error(self, message):
        """Log error message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.error(safe_message)
        self.log_to_db('ERROR', safe_message)
    
    def log_to_db(self, level, message):
        """Log message to database via PHP API (optional - won't crash if API unavailable)"""
        try:
            # Ensure message is ASCII-safe
            safe_message = str(message).encode('ascii', 'replace').decode('ascii')
            
            # Use PHP API instead of direct database connection
            import requests
            php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')
            url = f"{php_api_base}/add_log.php"
            
            data = {
                'job_id': self.job_id,
                'level': level,
                'message': safe_message[:1000]  # Limit length
            }
            
            # Non-blocking: don't wait too long for API response
            requests.post(url, json=data, timeout=2)
        except Exception:
            # Silently fail - API logging is optional
            # Training continues even if logging fails
            pass

class EnhancedPestDataset(Dataset):
    """Enhanced dataset with better error handling and statistics"""
    
    def __init__(self, data_dir, transform=None, logger=None, classes_from_yaml=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.logger = logger
        self.samples = []
        self.class_counts = {}
        self.classes_from_yaml = classes_from_yaml  # Classes from data.yaml
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset with statistics"""
        if not self.data_dir.exists():
            if self.logger:
                self.logger.error(f"Dataset directory not found: {self.data_dir}")
            return
        
        # PRIORITY: Use classes from YAML if provided, otherwise detect from directories
        if self.classes_from_yaml:
            self.classes = self.classes_from_yaml.copy()
            self.classes.sort()
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            if self.logger:
                self.logger.info(f"Using classes from data.yaml: {self.classes}")
        else:
            # Fallback: Get pest classes from directory structure
            self.classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            self.classes.sort()
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            if self.logger:
                self.logger.info(f"Found classes from directory structure: {self.classes}")
        
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
            # Use ASCII-safe characters for Windows compatibility
            progress_bar_length = 30
            filled = int(progress_bar_length * batch_progress / 100)
            bar = '#' * filled + '-' * (progress_bar_length - filled)  # ASCII-safe progress bar
            
            # Format like YOLOv8: epoch/batch  loss  accuracy  progress_bar
            progress_line = f"  {batch_idx+1}/{len(dataloader)}  {loss.item():.4f}  {current_acc:.1f}%  [{bar}] {batch_progress:.0f}%"
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
        """Save training metrics to database (optional - only if pymysql available)"""
        if not PYMYSQL_AVAILABLE or not DB_CONFIG:
            # Skip if pymysql not available (we're on Heroku using PHP API)
            return
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
        self.logger.info(f"Dataset contains {num_classes} classes: {train_dataset.classes}")
        print(f"Number of classes detected: {num_classes}", flush=True)
        print(f"Classes: {train_dataset.classes}", flush=True)
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
    
    def convert_to_onnx(self, model, model_path, input_size=(3, 224, 224)):
        """Convert PyTorch model to ONNX format"""
        try:
            onnx_path = model_path.with_suffix('.onnx')
            
            self.logger.info(f"Converting model to ONNX format...")
            print(f"[INFO] Converting model to ONNX...", flush=True)
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_size)
            
            # Export to ONNX (torch.onnx.export is built into PyTorch)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Validate ONNX model (optional)
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                self.logger.info(f"[OK] ONNX model validated successfully")
            except ImportError:
                # onnx package not available, skip validation
                pass
            except Exception as e:
                self.logger.warning(f"ONNX validation warning: {e}")
            
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"[OK] ONNX model saved to: {onnx_path} ({onnx_size_mb:.2f} MB)")
            print(f"[OK] ONNX model saved: {onnx_path.name} ({onnx_size_mb:.2f} MB)", flush=True)
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to ONNX: {e}")
            print(f"[ERROR] ONNX conversion failed: {e}", flush=True)
            return None
    
    def upload_model_to_server(self, model_path, accuracy, model_type='onnx'):
        """Upload model file to web server via PHP API with retry logic and SSL error handling"""
        import requests
        import base64
        import time
        
        # Try to import retry utilities (optional)
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            RETRY_AVAILABLE = True
        except ImportError:
            RETRY_AVAILABLE = False
            self.logger.warning("urllib3 retry not available, using simple retry logic")
        
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')
        upload_url = f"{php_api_base}/upload_model.php"
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False
        
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Uploading model to server... (Size: {model_size_mb:.2f} MB)")
        print(f"[INFO] Uploading model to server... ({model_size_mb:.2f} MB)", flush=True)
        
        # Read model file - use multipart/form-data for better efficiency with large files
        try:
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            # For large files, use multipart/form-data instead of base64 JSON
            # This is more memory efficient and faster
            self.logger.info(f"Model file read ({model_size_mb:.2f} MB)")
        except Exception as e:
            self.logger.error(f"Failed to read model file: {e}")
            return False
        
        # Prepare upload data - use files parameter for multipart upload
        # This is more efficient than base64 encoding in JSON
        upload_data = {
            'job_id': str(self.job_id),
            'accuracy': str(accuracy),
            'model_type': model_type,
            'model_size_mb': str(model_size_mb)
        }
        
        # Prepare file for multipart upload
        files = {
            'model_file': (model_path.name, model_bytes, 'application/octet-stream')
        }
        
        # Create session with retry adapter if available
        if RETRY_AVAILABLE:
            retry_strategy = Retry(
                total=3,
                backoff_factor=2,  # Wait 2, 4, 8 seconds between retries
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        else:
            session = requests.Session()
        
        # Upload with retries and better SSL handling
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"Upload attempt {attempt}/{max_attempts}...")
                print(f"[INFO] Upload attempt {attempt}/{max_attempts}...", flush=True)
                
                # Use multipart/form-data for large files (more efficient than JSON base64)
                # Use longer timeout for large files (10 minutes)
                response = session.post(
                    upload_url,
                    data=upload_data,
                    files=files,
                    timeout=600,  # 10 min timeout for large models
                    verify=True,  # Verify SSL certificate
                    stream=False  # Don't stream, send all at once
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        model_info = result.get('model', {})
                        self.logger.info(f"[OK] Model uploaded successfully: {model_info.get('version', 'N/A')}")
                        self.logger.info(f"  Path: {model_info.get('path', 'N/A')}")
                        self.logger.info(f"  Size: {model_info.get('size_mb', 0):.2f} MB")
                        self.logger.info(f"  Accuracy: {accuracy:.2f}%")
                        print(f"[OK] Model uploaded and activated: {model_info.get('version', 'N/A')}", flush=True)
                        return True
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        self.logger.error(f"Upload failed: {error_msg}")
                        if attempt < max_attempts:
                            wait_time = 2 ** attempt
                            print(f"[WARN] Retrying in {wait_time} seconds...", flush=True)
                            time.sleep(wait_time)
                            continue
                        return False
                else:
                    error_text = response.text[:500] if response.text else "No error message"
                    self.logger.error(f"Upload failed: HTTP {response.status_code} - {error_text}")
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt
                        print(f"[WARN] HTTP {response.status_code}, retrying in {wait_time} seconds...", flush=True)
                        time.sleep(wait_time)
                        continue
                    return False
                    
            except requests.exceptions.SSLError as e:
                self.logger.warning(f"SSL error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] SSL error, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"SSL error after {max_attempts} attempts: {e}")
                    print(f"[ERROR] SSL connection failed after {max_attempts} attempts", flush=True)
                    return False
                    
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Timeout, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Timeout after {max_attempts} attempts")
                    print(f"[ERROR] Upload timeout after {max_attempts} attempts", flush=True)
                    return False
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Connection error, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Connection error after {max_attempts} attempts: {e}")
                    print(f"[ERROR] Connection failed after {max_attempts} attempts", flush=True)
                    return False
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Error occurred, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Failed to upload model after {max_attempts} attempts: {e}")
                    print(f"[ERROR] Model upload error: {e}", flush=True)
                    return False
        
        return False
    
    def save_model(self, model, accuracy, classes):
        """Save model to database and file system - Auto-activates new model"""
        try:
            # Create model directory (in parent directory, same as root)
            script_dir = Path(__file__).resolve().parent
            model_dir = script_dir / "models" / f"job_{self.job_id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model file
            model_path = model_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Get model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(f"[OK] PyTorch model saved to: {model_path} ({model_size_mb:.2f} MB)")
            print(f"[OK] Model saved: {model_path.name} ({model_size_mb:.2f} MB)", flush=True)
            
            # Convert to ONNX for inference
            onnx_path = self.convert_to_onnx(model, model_path)
            
            # Upload to server (ONNX preferred, fallback to PyTorch)
            upload_success = False
            if onnx_path and onnx_path.exists():
                upload_success = self.upload_model_to_server(onnx_path, accuracy, 'onnx')
            else:
                # Fallback: upload PyTorch model
                self.logger.info("Uploading PyTorch model (ONNX conversion not available)")
                upload_success = self.upload_model_to_server(model_path, accuracy, 'pth')
            
            # Save to database (only if pymysql available - local development)
            if PYMYSQL_AVAILABLE and DB_CONFIG:
                try:
                    conn = pymysql.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    
                    # Generate version number
                    cursor.execute("SELECT MAX(CAST(SUBSTRING(version, 2) AS UNSIGNED)) FROM model_versions")
                    result = cursor.fetchone()
                    next_version = (result[0] or 0) + 1
                    
                    # Deactivate all previous models
                    cursor.execute("UPDATE model_versions SET is_active = 0, is_current = 0")
                    
                    # Use uploaded path if available, otherwise local path
                    model_db_path = onnx_path.name if onnx_path else model_path.name
                    
                    # Insert new model as active
                    cursor.execute(
                        "INSERT INTO model_versions (version, model_path, accuracy, training_job_id, model_size_mb, is_active, is_current, deployed_at) VALUES (%s, %s, %s, %s, %s, 1, 1, NOW())",
                        (f"v{next_version}", model_db_path, accuracy/100, self.job_id, model_size_mb)
                    )
                    
                    conn.commit()
                    conn.close()
                    
                    self.logger.info(f"[OK] Model v{next_version} saved and automatically activated with accuracy: {accuracy:.2f}%")
                    self.logger.info(f"[INFO] All previous models have been deactivated")
                except Exception as e:
                    self.logger.warning(f"Could not save model to database: {e}")
            
            if upload_success:
                self.logger.info(f"[OK] Model uploaded to server and activated successfully!")
                print(f"[OK] Model is now active and ready for detection!", flush=True)
            else:
                self.logger.warning(f"[WARNING] Model saved locally but upload to server failed")
                print(f"[WARNING] Model saved but not uploaded - check server connection", flush=True)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            print(f"[ERROR] Failed to save model: {e}", flush=True)

def update_job_status(job_id, status, error_message=None):
    """Update training job status via PHP API"""
    try:
        import requests
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')
        url = f"{php_api_base}/update_status.php"
        
        data = {
            'job_id': job_id,
            'status': status
        }
        if error_message:
            data['message'] = str(error_message).encode('ascii', 'replace').decode('ascii')[:500]
        
        requests.post(url, json=data, timeout=5)
    except Exception:
        # Silently fail - API updates are optional
        pass

def load_classes_from_yaml(yaml_path):
    """Load class names from data.yaml file"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # Try to get names from yaml
        if 'names' in yaml_data:
            names = yaml_data['names']
            # Handle both list and dict formats
            if isinstance(names, list):
                # Convert display names back to class codes (lowercase, replace spaces with underscores)
                class_codes = [name.lower().replace(' ', '_').replace('-', '_') for name in names]
                return class_codes
            elif isinstance(names, dict):
                # If it's a dict like {0: 'class1', 1: 'class2'}, get values
                class_codes = [name.lower().replace(' ', '_').replace('-', '_') for name in names.values()]
                return class_codes
        
        return None
    except Exception as e:
        print(f"Warning: Could not load classes from {yaml_path}: {e}", flush=True)
        return None

def download_dataset_from_server(script_dir, logger):
    """Download organized dataset from web server if not available locally"""
    try:
        import requests
        import zipfile
        import tempfile
        
        # Check if dataset already exists locally
        organized_dir = script_dir / "training_data" / "dataset_organized"
        if organized_dir.exists() and (organized_dir / "data.yaml").exists():
            print("[INFO] Dataset already exists locally, skipping download", flush=True)
            return True
        
        # Get PHP API base URL
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/Proto1/api/training')
        download_url = f"{php_api_base}/download_dataset.php?format=zip"
        
        print(f"[INFO] Downloading dataset from server...", flush=True)
        print(f"  URL: {download_url}", flush=True)
        logger.info(f"Downloading dataset from {download_url}")
        
        # Download ZIP file
        response = requests.get(download_url, timeout=300, stream=True)  # 5 min timeout for large files
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to download dataset: HTTP {response.status_code}", flush=True)
            logger.error(f"Dataset download failed: HTTP {response.status_code}")
            if response.status_code == 404:
                print("[ERROR] Dataset not found on server. Please upload a dataset first.", flush=True)
            return False
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = tmp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
        
        print(f"[OK] Dataset downloaded ({os.path.getsize(tmp_zip_path) / 1024 / 1024:.1f} MB)", flush=True)
        logger.info(f"Dataset downloaded: {os.path.getsize(tmp_zip_path) / 1024 / 1024:.1f} MB")
        
        # Extract ZIP
        print("[INFO] Extracting dataset...", flush=True)
        logger.info("Extracting dataset")
        
        # Create organized_dir directly (where we want the dataset)
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Check ZIP structure first
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Check if ZIP has a root folder
            if file_list:
                first_path = file_list[0]
                if '/' in first_path:
                    root_folder = first_path.split('/')[0]
                    print(f"[INFO] ZIP contains root folder: {root_folder}", flush=True)
                    
                    # If ZIP has 'dataset_organized' as root, extract to parent and let it create the folder
                    if root_folder == 'dataset_organized' or 'dataset' in root_folder.lower():
                        training_data_dir = script_dir / "training_data"
                        training_data_dir.mkdir(parents=True, exist_ok=True)
                        zip_ref.extractall(training_data_dir)
                        print(f"[INFO] Extracted to {training_data_dir}, dataset should be at {organized_dir}", flush=True)
                    else:
                        # Extract directly to organized_dir
                        zip_ref.extractall(organized_dir)
                        print(f"[INFO] Extracted directly to {organized_dir}", flush=True)
                else:
                    # Files are at root of ZIP, extract directly to organized_dir
                    zip_ref.extractall(organized_dir)
                    print(f"[INFO] Extracted root files to {organized_dir}", flush=True)
            else:
                print("[ERROR] ZIP file is empty", flush=True)
                return False
        
        # Clean up temporary ZIP
        os.unlink(tmp_zip_path)
        
        # Verify extraction - check multiple possible locations
        possible_locations = [
            organized_dir / "data.yaml",  # Expected location
            script_dir / "training_data" / "dataset_organized" / "data.yaml",  # If extracted with root folder
        ]
        
        # Also check if dataset_organized was created as a subfolder
        training_data_dir = script_dir / "training_data"
        if training_data_dir.exists():
            for item in training_data_dir.iterdir():
                if item.is_dir() and "dataset" in item.name.lower():
                    yaml_path = item / "data.yaml"
                    if yaml_path.exists():
                        possible_locations.append(yaml_path)
                        # If found here, update organized_dir reference
                        organized_dir = item
                        print(f"[INFO] Found dataset in subfolder: {organized_dir}", flush=True)
        
        data_yaml_path = None
        for loc in possible_locations:
            if loc.exists():
                data_yaml_path = loc
                print(f"[OK] Found data.yaml at: {data_yaml_path}", flush=True)
                break
        
        if data_yaml_path and (organized_dir / "data.yaml").exists():
            print(f"[OK] Dataset extracted successfully to {organized_dir}", flush=True)
            logger.info(f"Dataset extracted to {organized_dir}")
            return True
        else:
            print("[ERROR] Dataset extracted but data.yaml not found", flush=True)
            print(f"[DEBUG] Checked locations:", flush=True)
            for loc in possible_locations:
                exists = "✓" if loc.exists() else "✗"
                print(f"  {exists} {loc}", flush=True)
            # List what was actually extracted
            if training_data_dir.exists():
                print(f"[DEBUG] Contents of {training_data_dir}:", flush=True)
                for item in training_data_dir.iterdir():
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})", flush=True)
            logger.error("Dataset extracted but data.yaml not found")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error downloading dataset: {e}", flush=True)
        logger.error(f"Network error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}", flush=True)
        logger.error(f"Error downloading dataset: {e}")
        return False

def create_combined_dataset(logger):
    """Create combined dataset from original and collected data"""
    import sys
    print("Creating combined dataset...", flush=True)
    logger.info("Creating combined dataset...")
    
    # Directories - use parent directory (Proto1/)
    script_dir = Path(__file__).resolve().parent
    
    # PRIORITY 0: Download dataset from server if not available locally (for Heroku)
    organized_dir = script_dir / "training_data" / "dataset_organized"
    if not organized_dir.exists() or not (organized_dir / "data.yaml").exists():
        print("[INFO] Dataset not found locally, attempting to download from server...", flush=True)
        logger.info("Dataset not found locally, attempting download from server")
        download_success = download_dataset_from_server(script_dir, logger)
        if not download_success:
            print("[WARNING] Could not download dataset from server, will try fallback methods...", flush=True)
            logger.warning("Could not download dataset from server, trying fallback methods")
    
    # PRIORITY 1: Check for organized dataset from smart import (has data.yaml)
    organized_dir = script_dir / "training_data" / "dataset_organized"
    organized_yaml = organized_dir / "data.yaml"
    
    # Always try to load classes from data.yaml first (from organized dataset created during import)
    pest_classes = load_classes_from_yaml(organized_yaml)
    
    if organized_yaml.exists():
        if pest_classes:
            print(f"[OK] Found data.yaml from import", flush=True)
            print(f"  Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}", flush=True)
            logger.info(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}")
        else:
            print(f"Warning: data.yaml exists but could not parse classes", flush=True)
            logger.warning("data.yaml exists but could not parse classes")
    
    # Check if organized dataset exists (from smart import)
    organized_train_images = organized_dir / "train" / "images"
    organized_train_labels = organized_dir / "train" / "labels"
    organized_val_images = organized_dir / "valid" / "images"
    organized_val_labels = organized_dir / "valid" / "labels"
    
    # PRIORITY 1A: Use database to get images (most reliable - images are stored with pest_class)
    if organized_yaml.exists() and pest_classes:
        print(f"[INFO] Checking database for imported images...", flush=True)
        logger.info("Checking database for imported images")
        
        # Only try database if pymysql is available (local development)
        if not PYMYSQL_AVAILABLE or not DB_CONFIG:
            print("[INFO] Skipping database image lookup (using PHP API gateway)", flush=True)
            db_images = []
        else:
            try:
                conn = pymysql.connect(**DB_CONFIG)
                cursor = conn.cursor()
                
                # Get all images from training_images table
                cursor.execute("SELECT file_path, pest_class FROM training_images WHERE is_verified = 1")
                db_images = cursor.fetchall()
                conn.close()
            except Exception as e:
                print(f"[WARNING] Could not access database for images: {e}", flush=True)
                db_images = []
        
        if db_images and len(db_images) > 0:
            print(f"[OK] Found {len(db_images)} images in database", flush=True)
            logger.info(f"Found {len(db_images)} images in database")
            
            # Create classification-ready dataset structure
            classification_train_dir = organized_dir / "classification" / "train"
            classification_val_dir = organized_dir / "classification" / "val"
            
            # Create class folders
            for split_dir in [classification_train_dir, classification_val_dir]:
                split_dir.mkdir(parents=True, exist_ok=True)
                for class_name in pest_classes:
                    (split_dir / class_name).mkdir(exist_ok=True)
            
            # Normalize pest_class names to match YAML classes (handle variations)
            def normalize_class_name(db_class):
                """Normalize database class name to match YAML class names"""
                db_class_lower = db_class.lower().replace(' ', '_').replace('-', '_')
                # Try exact match first
                for yaml_class in pest_classes:
                    if db_class_lower == yaml_class.lower():
                        return yaml_class
                # Try partial match
                for yaml_class in pest_classes:
                    if yaml_class.lower() in db_class_lower or db_class_lower in yaml_class.lower():
                        return yaml_class
                return None
            
            # Reorganize images from database
            import random
            random.seed(42)
            train_count = 0
            val_count = 0
            
            for file_path, db_pest_class in db_images:
                img_path = Path(file_path)
                if not img_path.exists():
                    # Try relative path from script directory
                    img_path = script_dir / file_path.lstrip('/')
                    if not img_path.exists():
                        continue
                
                # Normalize class name
                class_name = normalize_class_name(db_pest_class)
                if not class_name:
                    logger.warning(f"Could not map database class '{db_pest_class}' to YAML classes")
                    continue
                
                # Split 80% train, 20% val
                is_train = random.random() < 0.8
                dest_dir = classification_train_dir if is_train else classification_val_dir
                dest = dest_dir / class_name / img_path.name
                
                if not dest.exists():
                    try:
                        shutil.copy2(img_path, dest)
                        if is_train:
                            train_count += 1
                        else:
                            val_count += 1
                    except Exception as e:
                        logger.warning(f"Error copying {img_path}: {e}")
                        continue
            
            if train_count > 0 or val_count > 0:
                print(f"[OK] Reorganized {len(db_images)} images from database: {train_count} train, {val_count} val", flush=True)
                logger.info(f"Reorganized {len(db_images)} images from database: {train_count} train, {val_count} val")
                print(f"  Using reorganized dataset at: {classification_train_dir}", flush=True)
                sys.stdout.flush()
                return classification_train_dir, classification_val_dir, pest_classes
            else:
                print(f"[WARN] Could not copy images from database, trying YOLO labels...", flush=True)
                logger.warning("Could not copy images from database, trying YOLO labels")
        else:
            print(f"[INFO] No images found in database, trying YOLO format...", flush=True)
            logger.info("No images found in database, trying YOLO format")
    
    # PRIORITY 1B: Reorganize from YOLO format (if database method didn't work)
    if organized_yaml.exists() and organized_train_images.exists() and pest_classes:
        print(f"[INFO] Found organized dataset from import, reorganizing from YOLO format...", flush=True)
        logger.info("Reorganizing organized dataset from YOLO format into class folders")
        
        # Create classification-ready dataset structure
        classification_train_dir = organized_dir / "classification" / "train"
        classification_val_dir = organized_dir / "classification" / "val"
        
        # Create class folders
        for split_dir in [classification_train_dir, classification_val_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in pest_classes:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        # Function to reorganize images based on YOLO labels
        def reorganize_from_yolo(images_dir, labels_dir, output_dir, split_name):
            if not images_dir.exists():
                print(f"  [WARN] Images directory not found: {images_dir}", flush=True)
                return 0
            if not labels_dir.exists():
                print(f"  [WARN] Labels directory not found: {labels_dir}", flush=True)
                return 0
            
            reorganized_count = 0
            skipped_no_label = 0
            skipped_invalid_label = 0
            skipped_duplicate = 0
            skipped_invalid_class = 0
            
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
            print(f"  [INFO] Found {len(images)} images in {images_dir}", flush=True)
            
            for img_path in images:
                label_path = labels_dir / (img_path.stem + '.txt')
                if not label_path.exists():
                    skipped_no_label += 1
                    continue
                
                # Read first line of label to get class index
                try:
                    with open(label_path, 'r') as f:
                        first_line = f.readline().strip()
                        if not first_line:
                            skipped_invalid_label += 1
                            continue
                        
                        parts = first_line.split()
                        if len(parts) < 5:
                            skipped_invalid_label += 1
                            continue
                        
                        class_index = int(parts[0])
                        # Map class index to class name (assuming indices match YAML order)
                        if class_index >= len(pest_classes):
                            skipped_invalid_class += 1
                            continue
                        
                        class_name = pest_classes[class_index]
                        # Copy image to class folder
                        dest = output_dir / class_name / img_path.name
                        if dest.exists():  # Avoid duplicates
                            skipped_duplicate += 1
                            continue
                        
                        shutil.copy2(img_path, dest)
                        reorganized_count += 1
                except Exception as e:
                    skipped_invalid_label += 1
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            # Print detailed statistics
            total_processed = len(images)
            total_skipped = skipped_no_label + skipped_invalid_label + skipped_duplicate + skipped_invalid_class
            print(f"  [INFO] {split_name}: {reorganized_count}/{total_processed} images reorganized", flush=True)
            if total_skipped > 0:
                print(f"  [INFO]   Skipped: {skipped_no_label} (no label), {skipped_invalid_label} (invalid label), {skipped_duplicate} (duplicate), {skipped_invalid_class} (invalid class)", flush=True)
            
            return reorganized_count
        
        # Reorganize train and val sets
        train_count = reorganize_from_yolo(organized_train_images, organized_train_labels, classification_train_dir, "train")
        val_count = reorganize_from_yolo(organized_val_images, organized_val_labels, classification_val_dir, "val")
        
        # Also include test images in training set (optional - use all available data)
        organized_test_images = organized_dir / "test" / "images"
        organized_test_labels = organized_dir / "test" / "labels"
        test_count = 0
        if organized_test_images.exists() and organized_test_labels.exists():
            # Add test images to training set (not validation, to maximize training data)
            test_count = reorganize_from_yolo(organized_test_images, organized_test_labels, classification_train_dir, "test")
            if test_count > 0:
                print(f"[INFO] Added {test_count} test images to training set (using all available data)", flush=True)
                logger.info(f"Added {test_count} test images to training set")
        
        if train_count > 0 or val_count > 0:
            total_train = train_count + test_count
            print(f"[OK] Reorganized dataset from YOLO: {total_train} train images ({train_count} train + {test_count} test), {val_count} val images", flush=True)
            logger.info(f"Reorganized dataset from YOLO: {total_train} train images, {val_count} val images")
            print(f"  Total images used: {total_train + val_count} (out of {train_count + val_count + test_count} available)", flush=True)
            print(f"  Using reorganized dataset at: {classification_train_dir}", flush=True)
            sys.stdout.flush()
            return classification_train_dir, classification_val_dir, pest_classes
        else:
            print(f"[WARN] No images found in organized dataset, falling back...", flush=True)
            logger.warning("No images found in organized dataset, falling back")
    
    # PRIORITY 2: Fallback to old dataset structure (only if organized dataset doesn't exist or has no images)
    original_train_dir = script_dir / "ml_training" / "datasets" / "processed" / "train"
    original_val_dir = script_dir / "ml_training" / "datasets" / "processed" / "val"
    collected_data_dir = script_dir / "ml_training" / "datasets" / "auto_collected"
    combined_dir = script_dir / "ml_training" / "datasets" / "combined"
    
    # Fallback to hardcoded classes if yaml not found
    if not pest_classes:
        print("Warning: data.yaml not found, using default classes", flush=True)
        logger.warning("data.yaml not found, using default classes")
        pest_classes = ['leptocorisa_oratorius', 'nephotettix_virescens', 'nilaparvata_lugens', 'scotinophara_coarctata', 'scirpophaga_incertulas']
    else:
        print(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}", flush=True)
        logger.info(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}")
    
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
            return original_train_dir, original_val_dir, pest_classes
        else:
            print(f"[WARN] Val dir missing, using train for both", flush=True)
            return original_train_dir, original_train_dir, pest_classes
    
    # Only create combined dataset if original doesn't exist
    print("Creating combined dataset structure...", flush=True)
    combined_train_dir = combined_dir / "train"
    combined_val_dir = combined_dir / "val"
    
    for split_dir in [combined_train_dir, combined_val_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in pest_classes:
            (split_dir / class_name).mkdir(exist_ok=True)
    
    # Copy original training data
    logger.info("Copying original training data...")
    for class_name in pest_classes:
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
    
    for class_name in pest_classes:
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
    
    return combined_train_dir, combined_val_dir, pest_classes

# YOLO training function removed - reverting to classification

def main():
    """Main training function - Classification using ResNet18"""
    parser = argparse.ArgumentParser(description='Admin Training Script - YOLO Object Detection')
    parser.add_argument('--job_id', type=int, required=True, help='Training job ID')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (not used for YOLO)')
    
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
            train_dir, val_dir, classes_from_yaml = create_combined_dataset(logger)
            print(f"[OK] Dataset created: Train={train_dir}, Val={val_dir}", flush=True)
            if classes_from_yaml:
                print(f"[OK] Classes from data.yaml: {len(classes_from_yaml)} classes - {classes_from_yaml}", flush=True)
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
        print(f"  Train directory: {train_dir}", flush=True)
        print(f"  Val directory: {val_dir}", flush=True)
        if classes_from_yaml:
            print(f"  Classes from data.yaml: {len(classes_from_yaml)} classes - {classes_from_yaml}", flush=True)
        sys.stdout.flush()
        logger.info("Loading datasets...")
        logger.info(f"Train directory: {train_dir}")
        logger.info(f"Val directory: {val_dir}")
        if classes_from_yaml:
            logger.info(f"Classes from data.yaml: {classes_from_yaml}")
        
        # Pass classes from YAML to dataset class so it uses the correct number of classes
        train_dataset = EnhancedPestDataset(train_dir, transform=train_transforms, logger=logger, classes_from_yaml=classes_from_yaml)
        val_dataset = EnhancedPestDataset(val_dir, transform=val_transforms, logger=logger, classes_from_yaml=classes_from_yaml)
        
        print(f"[OK] Datasets loaded: Train={len(train_dataset)} samples, Val={len(val_dataset)} samples", flush=True)
        print(f"[INFO] Number of classes detected: {len(train_dataset.classes)}", flush=True)
        print(f"[INFO] Classes: {train_dataset.classes}", flush=True)
        sys.stdout.flush()
        
        # Log dataset statistics
        train_stats = train_dataset.get_statistics()
        val_stats = val_dataset.get_statistics()
        
        logger.info(f"Training dataset: {train_stats}")
        logger.info(f"Validation dataset: {val_stats}")
        logger.info(f"Number of classes: {len(train_dataset.classes)}")
        logger.info(f"Classes: {train_dataset.classes}")
        
        # Start training
        model = trainer.train(train_dataset, val_dataset)
        
        # After training completes, ensure final model is uploaded
        if trainer.best_accuracy > 0:
            logger.info(f"Training completed! Final best accuracy: {trainer.best_accuracy:.2f}%")
            print(f"[OK] Training completed! Best accuracy: {trainer.best_accuracy:.2f}%", flush=True)
            
            # The model should already be saved and uploaded during training (when best model found)
            # But let's verify and upload final model if needed
            script_dir = Path(__file__).resolve().parent
            model_dir = script_dir / "models" / f"job_{args.job_id}"
            onnx_path = model_dir / "best_model.onnx"
            pth_path = model_dir / "best_model.pth"
            
            # Check if model was already uploaded, if not upload now
            if onnx_path.exists():
                logger.info("Verifying final model upload...")
                # Model should already be uploaded, but log success
                print(f"[OK] Model ready: {onnx_path.name}", flush=True)
            elif pth_path.exists():
                # Only PyTorch model exists, try to convert and upload
                logger.info("Converting final model to ONNX and uploading...")
                onnx_path = trainer.convert_to_onnx(model, pth_path)
                if onnx_path and onnx_path.exists():
                    trainer.upload_model_to_server(onnx_path, trainer.best_accuracy, 'onnx')
        
        # Update job status to completed
        update_job_status(args.job_id, 'completed')
        logger.info(f"Training job {args.job_id} completed successfully!")
        print(f"[OK] Training job {args.job_id} completed!", flush=True)
        
    except Exception as e:
        # Ensure error message is ASCII-safe
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        logger.error(f"Training failed: {error_msg}")
        update_job_status(args.job_id, 'failed', error_msg)  # Use ASCII-safe error message
        sys.exit(1)

if __name__ == "__main__":
    main()
