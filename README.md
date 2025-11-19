# 🌾 AgriShield Flask ML API - Deployment Guide

## 📋 **What This Is:**

A complete Flask ML API system with 3 ML components:
1. **Pest Detection** - YOLO object detection
2. **Pest Forecasting** - 7-day risk prediction
3. **Admin Training** - Model training system

---

## ✅ **Pre-Deployment Checklist:**

### **Required Files (All Present):**
- ✅ `app.py` - Main Flask application
- ✅ `admin_training_script.py` - Training script
- ✅ `config.py` - Configuration (reads from parent config.php)
- ✅ `requirements.txt` - Python dependencies
- ✅ `run_auto.sh` - Auto-start script
- ✅ `start_flask.sh` - Startup script
- ✅ `start_flask.bat` - Windows startup script
- ✅ `env_template.txt` - Environment variables template

### **Configuration:**
- ✅ Reads database credentials from `../config.php` automatically
- ✅ No hardcoded values
- ✅ All configurable via environment variables

---

## 🚀 **Quick Deployment Steps:**

### **1. Upload Files:**
Upload entire `AgriShield_ML_Flask` folder to server.

### **2. Set Permissions:**
```bash
cd AgriShield_ML_Flask
chmod +x run_auto.sh
chmod +x start_flask.sh
chmod +x app.py
chmod +x admin_training_script.py
```

### **3. Install Dependencies:**
```bash
cd AgriShield_ML_Flask
pip3 install -r requirements.txt
```

### **4. Place ML Model:**
Place `best.pt` model file in one of these locations:
- `AgriShield_ML_Flask/ml_models/pest_detection/best.pt` (preferred)
- `AgriShield_ML_Flask/models/best.pt`
- `AgriShield_ML_Flask/best.pt`

### **5. Start Flask:**
```bash
./run_auto.sh start
```

### **6. Auto-Start on Boot (Optional):**
```bash
crontab -e
# Add: @reboot cd /path/to/Proto1/AgriShield_ML_Flask && ./run_auto.sh start
```

---

## 🔧 **Configuration:**

### **Database:**
- Automatically reads from `../config.php` (parent directory)
- No manual configuration needed
- Credentials: Already configured in config.php

### **Port:**
- Default: `8000`
- Changeable via environment variable: `FLASK_PORT=8000`

### **Host:**
- Default: `0.0.0.0` (all interfaces)
- Changeable via environment variable: `FLASK_HOST=0.0.0.0`

---

## 📡 **API Endpoints:**

### **Pest Detection:**
- `POST /detect/` - Detect pests in image
- `POST /classify/` - Mobile app compatibility

### **Pest Forecasting:**
- `GET /forecast/` - 7-day pest forecast
- `GET /forecast/quick/` - Quick forecast
- `GET /forecast/current/` - Current forecast
- `GET /forecast/update/` - Manual weather update

### **Health Check:**
- `GET /health/` - Health check
- `GET /status/` - Status check

---

## 🧪 **Test After Deployment:**

```bash
# Test health endpoint
curl http://localhost:8000/health/

# Test detection (if you have an image)
curl -X POST -F "image=@test.jpg" http://localhost:8000/detect/

# Check if running
ps aux | grep app.py
```

---

## 📝 **Important Notes:**

1. **Model File:** Make sure `best.pt` is included or placed in correct location
2. **Database:** Ensure `config.php` exists in parent directory with correct credentials
3. **Port:** Make sure port 8000 is not blocked by firewall
4. **Python:** Requires Python 3.8+
5. **Dependencies:** All listed in `requirements.txt`

---

## 🔍 **Troubleshooting:**

### **Flask won't start:**
- Check Python version: `python3 --version`
- Check dependencies: `pip3 list`
- Check logs: `tail -f flask_auto.log`

### **Model not found:**
- Place `best.pt` in `ml_models/pest_detection/` folder
- Check model path in logs

### **Database connection failed:**
- Verify `config.php` exists in parent directory
- Check database credentials in `config.php`
- Test connection: `python3 -c "from config import DB_CONFIG; print(DB_CONFIG)"`

---

## 📊 **System Requirements:**

- Python 3.8+
- MySQL/MariaDB database
- ~2GB RAM (for YOLO model)
- Port 8000 available
- Internet connection (for weather API)

---

## ✅ **Ready for Deployment!**

All files are present and configured. Just upload, install dependencies, and start!

---

## 📞 **Support:**

See `SERVER_SETUP.md` for detailed setup instructions.
See `AUTO_START_GUIDE.md` for auto-start options.
See `ML_SYSTEMS_ANALYSIS.md` for system overview.

