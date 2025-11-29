# Heroku Training Service

## 🎯 Fully Independent Training on Heroku

This service runs training **completely independently** - no browser, no Colab, no manual steps!

---

## 🚀 Quick Deploy

### 1. Create Heroku App
```bash
heroku create your-training-service-name
```

### 2. Set Environment Variables
```bash
heroku config:set DB_HOST=auth-db1322.hstgr.io
heroku config:set DB_USER=u520834156_uAShield2025
heroku config:set DB_PASSWORD=:JqjB0@0zb6v
heroku config:set DB_NAME=u520834156_dbAgriShield
```

### 3. Deploy
```bash
git push heroku main
```

### 4. Update PHP
Edit `cloud_training_helper.php`:
```php
$training_service_url = "https://your-training-service-name.herokuapp.com";
```

---

## 📝 Files

- `app.py` - Flask training service
- `requirements.txt` - Python dependencies (CPU-only PyTorch for Heroku)
- `Procfile` - Heroku process file
- `.python-version` - Python 3.11
- `app.json` - Heroku app configuration

---

## ✅ How It Works

1. Admin clicks "Train in Cloud"
2. PHP creates training job in database
3. PHP calls Heroku service: `POST /train`
4. Service starts training in background
5. Training runs independently
6. Status updates in database automatically

**Fully automatic - no browser needed!** 🎯

---

## ⚠️ Important Notes

- **Heroku slug limit**: 500MB (use CPU-only PyTorch)
- **Training script**: Upload `admin_training_script.py` as `train.py`
- **Memory**: May need paid tier for large models
- **Timeout**: Training runs in background (no request timeout)

See `SETUP_HEROKU.md` for detailed setup instructions.

