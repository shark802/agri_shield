# ML Deployment Package

Complete ML deployment package for AgriShield Pest Detection System.

## 📦 Contents

### Core ML Components

1. **Detection API** (`pest_detection_api.py`)
   - ONNX Runtime-based pest detection
   - Optimized for Heroku (113 MB vs 528 MB)
   - Flask REST API

2. **Forecasting Engine** (`pest_forecasting_engine.py`)
   - Weather-based pest outbreak prediction
   - Uses scikit-learn (RandomForest, LinearRegression)
   - 7-day forecast generation

3. **Training Script** (`scripts/admin_training_script.py`)
   - ResNet18 classification model training
   - Database-integrated logging
   - Real-time progress monitoring

### Models

- **ONNX Models**: Located in `models/` directory
  - `best.onnx` - Main detection model
  - `best 2.onnx` - Alternative model
  - `best5.onnx` - Additional model

### Configuration

- **requirements.txt**: Heroku-optimized dependencies (ONNX Runtime)
- **Procfile**: Heroku deployment configuration
- **runtime.txt**: Python version (if needed)

### Documentation

- **HEROKU_DEPLOYMENT_GUIDE.md**: Complete deployment guide
- **HEROKU_DEPLOYMENT_ANALYSIS.md**: Size analysis and optimization
- **DEPENDENCY_ANALYSIS_REAL.md**: Real dependency sizes
- **CONVERSION_STATUS_FINAL.md**: TFLite conversion status

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run detection API
python pest_detection_api.py

# API will be available at http://localhost:5001
```

### Heroku Deployment

```bash
# Login to Heroku
heroku login

# Create app (if needed)
heroku create your-app-name

# Deploy
git add .
git commit -m "Deploy ML API"
git push heroku main

# Check status
heroku logs --tail
```

---

## 📊 API Endpoints

### Detection API

**POST** `/detect`
- Upload image for pest detection
- Returns: pest counts, recommendations, inference time

**GET** `/health`
- Health check endpoint
- Returns: model status, classes, framework info

### Example Request

```bash
curl -X POST http://localhost:5001/detect \
  -F "image=@test_image.jpg"
```

### Example Response

```json
{
  "status": "success",
  "pest_counts": {
    "Rice_Bug": 3,
    "green_hopper": 1,
    "brown_hopper": 0,
    "black-bug": 0
  },
  "recommendations": {
    "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin...",
    "green_hopper": "Imidacloprid or dinotefuran early..."
  },
  "inference_time_ms": 45.2,
  "model": "best.onnx",
  "framework": "ONNX Runtime"
}
```

---

## 📁 Folder Structure

```
ml_deployment/
├── pest_detection_api.py      # Main detection API (ONNX)
├── pest_forecasting_engine.py # Forecasting engine
├── requirements.txt            # Dependencies (Heroku-optimized)
├── Procfile                    # Heroku config
├── runtime.txt                 # Python version (optional)
├── models/                     # ONNX model files
│   ├── best.onnx
│   ├── best 2.onnx
│   └── best5.onnx
├── scripts/                    # Training scripts
│   └── admin_training_script.py
└── docs/                      # Documentation
    ├── HEROKU_DEPLOYMENT_GUIDE.md
    ├── HEROKU_DEPLOYMENT_ANALYSIS.md
    ├── DEPENDENCY_ANALYSIS_REAL.md
    └── CONVERSION_STATUS_FINAL.md
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Database (optional - can use config.php)
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=asdb

# API Port
PORT=5001

# Model Path (optional - auto-detected)
MODEL_PATH=models/best.onnx
```

### Model Path Priority

The API automatically searches for models in this order:
1. `models/best.onnx`
2. `models/best 2.onnx`
3. `models/best5.onnx`
4. `datasets/best.onnx` (fallback)

---

## 📈 Performance

### Size Comparison

| Component | PyTorch | ONNX Runtime | Savings |
|-----------|---------|-------------|---------|
| **Dependencies** | 528 MB | 113 MB | **79%** |
| **With Models** | 564 MB | 149 MB | **74%** |

### Inference Speed

- **ONNX Runtime**: ~45-60ms per image
- **Memory Usage**: ~50-100 MB RAM
- **Startup Time**: ~2-3 seconds

---

## 🛠️ Dependencies

### Production (Heroku)

- `onnxruntime==1.23.2` (41 MB)
- `flask==2.3.0` (2 MB)
- `opencv-python-headless==4.9.0.80` (30 MB)
- `numpy==1.24.3` (20 MB)
- `pillow==10.0.0` (5 MB)
- `pymysql==1.1.0` (2 MB)
- `gunicorn==21.2.0` (2 MB)

**Total: ~113 MB** (fits Heroku's 500 MB limit!)

### Development (Training)

- `torch==2.0.1` (400 MB)
- `torchvision==0.15.2` (50 MB)
- `ultralytics==8.0.196` (6 MB)
- `scikit-learn` (50 MB)
- Additional training dependencies

---

## 📝 Usage Examples

### Python Client

```python
import requests

# Detection
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/detect',
        files={'image': f}
    )
    result = response.json()
    print(result['pest_counts'])
```

### PHP Integration

```php
$ch = curl_init('http://localhost:5001/detect');
$cfile = new CURLFile($image_path, 'image/jpeg', 'image.jpg');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => ['image' => $cfile],
    CURLOPT_RETURNTRANSFER => true
]);
$response = curl_exec($ch);
$result = json_decode($response, true);
```

---

## 🔍 Troubleshooting

### Model Not Found

```bash
# Check model path
ls -lh models/

# Update MODEL_PATH environment variable
export MODEL_PATH=models/best.onnx
```

### ONNX Runtime Not Available

```bash
# Install ONNX Runtime
pip install onnxruntime==1.23.2

# Verify installation
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### Heroku Deployment Issues

1. **Slug Size**: Should be ~149 MB (well under 500 MB limit)
2. **Build Timeout**: ONNX Runtime installs quickly (~30 seconds)
3. **Memory**: Standard-1X dyno (512 MB) is sufficient

---

## 📚 Additional Resources

- **Deployment Guide**: See `docs/HEROKU_DEPLOYMENT_GUIDE.md`
- **Size Analysis**: See `docs/DEPENDENCY_ANALYSIS_REAL.md`
- **Conversion Status**: See `docs/CONVERSION_STATUS_FINAL.md`

---

## ✅ Checklist

Before deploying:

- [ ] ONNX models in `models/` directory
- [ ] Dependencies in `requirements.txt`
- [ ] `Procfile` configured correctly
- [ ] Environment variables set (if needed)
- [ ] Tested locally with `python pest_detection_api.py`
- [ ] Health check works: `curl http://localhost:5001/health`

---

## 🎯 Summary

**This package contains everything needed for ML deployment:**

✅ **Detection API** - ONNX Runtime (113 MB)  
✅ **Forecasting Engine** - scikit-learn  
✅ **Training Script** - PyTorch ResNet18  
✅ **Heroku Ready** - Optimized for 500 MB limit  
✅ **Documentation** - Complete guides  

**Ready to deploy!** 🚀

