# Heroku Deployment Analysis - ML Components

## Heroku Constraints

### Critical Limits:
- **Slug Size Limit:** 500 MB (hard limit)
- **Build Timeout:** 15 minutes
- **Memory:** 512 MB - 1 GB (depending on dyno type)
- **Disk:** Ephemeral (cleared on restart)

---

## Current Setup Analysis

### Current `requirements_heroku.txt`:
```
flask==3.0.0
torch==2.5.1+cpu
torchvision==0.20.1+cpu
ultralytics==8.3.203
opencv-python-headless==4.9.0.80
numpy==1.26.4
Pillow==10.2.0
pymysql==1.1.1
python-dotenv==1.0.1
requests==2.31.0
```

### Size Breakdown (Estimated):

| Package | Size | Notes |
|---------|------|-------|
| **torch** | ~400 MB | CPU-only version (smaller) |
| **torchvision** | ~50 MB | CPU-only |
| **ultralytics** | ~6 MB | |
| **opencv-python-headless** | ~30 MB | Headless (no GUI) |
| **numpy** | ~20 MB | |
| **flask** | ~2 MB | |
| **Other** | ~20 MB | |
| **Total** | **~528 MB** | ⚠️ **EXCEEDS 500 MB LIMIT!** |

---

## Problem: Current Setup Too Large!

### ❌ Current PyTorch Setup:
- **Total: ~528 MB**
- **Heroku Limit: 500 MB**
- **Status: EXCEEDS LIMIT!**

**Result:** Deployment will fail or be unreliable!

---

## Solution: Use ONNX Runtime

### ✅ ONNX Runtime Setup:

**New `requirements_heroku.txt`:**
```
flask==3.0.0
onnxruntime==1.23.2
opencv-python-headless==4.9.0.80
numpy==1.26.4
Pillow==10.2.0
pymysql==1.1.1
python-dotenv==1.0.1
requests==2.31.0
```

### Size Breakdown:

| Package | Size | Notes |
|---------|------|-------|
| **onnxruntime** | ~41 MB | Much smaller! |
| **opencv-python-headless** | ~30 MB | |
| **numpy** | ~20 MB | |
| **flask** | ~2 MB | |
| **Other** | ~20 MB | |
| **Total** | **~113 MB** | ✅ **WELL UNDER LIMIT!** |

---

## Comparison

| Setup | Size | Heroku Status | Savings |
|-------|------|---------------|---------|
| **Current (PyTorch)** | ~528 MB | ❌ **EXCEEDS LIMIT** | - |
| **ONNX Runtime** | ~113 MB | ✅ **FITS EASILY** | **79% smaller** |
| **Difference** | **415 MB saved** | | |

---

## Benefits for Heroku

### 1. ✅ Fits Slug Size Limit
- Current: 528 MB (exceeds 500 MB)
- ONNX: 113 MB (well under limit)
- **Room for growth:** 387 MB remaining

### 2. ✅ Faster Build Times
- Smaller packages = faster pip install
- Less download time
- Less extraction time

### 3. ✅ Faster Cold Starts
- Smaller runtime = faster loading
- Less memory usage
- Better performance

### 4. ✅ Lower Memory Usage
- ONNX Runtime: ~50-100 MB RAM
- PyTorch: ~200-300 MB RAM
- **More headroom for your app**

### 5. ✅ More Reliable
- Under size limit = no deployment failures
- Faster startup = better user experience
- Lower memory = fewer crashes

---

## Migration Steps

### Step 1: Update `requirements_heroku.txt`
Replace PyTorch with ONNX Runtime:
```txt
flask==3.0.0
onnxruntime==1.23.2
opencv-python-headless==4.9.0.80
numpy==1.26.4
Pillow==10.2.0
pymysql==1.1.1
python-dotenv==1.0.1
requests==2.31.0
```

### Step 2: Use ONNX API
- Switch from `pest_detection_api.py` (PyTorch)
- To `pest_detection_api_onnx.py` (ONNX Runtime)

### Step 3: Deploy ONNX Models
- Already have 16 ONNX files
- Upload to Heroku (or use S3/Cloud Storage)

### Step 4: Update Procfile (if exists)
```procfile
web: python pest_detection_api_onnx.py
```

---

## Model Storage Options

### Option 1: Include in Git (Small Models)
- ONNX files: ~12 MB each
- 3 main models: ~36 MB
- **Total: ~149 MB** (still under limit!)

### Option 2: External Storage (Recommended)
- Upload to AWS S3 / Google Cloud Storage
- Download on startup
- Better for multiple models

### Option 3: Heroku Buildpack
- Use buildpack for larger files
- More complex setup

---

## Final Recommendation

### ✅ Use ONNX Runtime for Heroku

**Why:**
1. ✅ **Fits 500 MB limit** (113 MB vs 528 MB)
2. ✅ **79% smaller** deployment
3. ✅ **Faster builds** and cold starts
4. ✅ **Lower memory** usage
5. ✅ **More reliable** deployment

**Action Items:**
1. Update `requirements_heroku.txt`
2. Switch to `pest_detection_api_onnx.py`
3. Deploy ONNX models
4. Test deployment

---

## Size Summary

| Component | PyTorch | ONNX Runtime | Savings |
|-----------|---------|--------------|---------|
| **Dependencies** | 528 MB | 113 MB | **415 MB** |
| **Models (3x)** | 36 MB | 36 MB | 0 MB |
| **Total** | **564 MB** | **149 MB** | **415 MB (74%)** |

**Result:** 
- ❌ PyTorch: **EXCEEDS 500 MB LIMIT**
- ✅ ONNX: **149 MB - FITS EASILY!**

---

## Conclusion

**For Heroku deployment, ONNX Runtime is ESSENTIAL!**

- Current setup **exceeds Heroku's 500 MB limit**
- ONNX Runtime **fits easily** with room to spare
- **79% size reduction**
- **Better performance** and reliability

**Switch to ONNX Runtime now for successful Heroku deployment!**

