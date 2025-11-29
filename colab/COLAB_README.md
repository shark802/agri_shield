# Google Colab Cloud Training Integration

## 🎯 Overview

This integration allows admins to train pest detection models in Google Colab (free GPU) directly from the admin interface at `https://agrishield.bccbsis.com`.

---

## 📁 Files

- **`AgriShield_Training_Colab.py`** - Python script to copy into Google Colab
- **`cloud_training_helper.php`** - PHP functions for cloud training
- **`GOOGLE_COLAB_SETUP.md`** - Detailed setup instructions
- **`database_migration.sql`** - Database schema updates
- **`README.md`** - This file

---

## 🚀 Quick Start

### 1. Database Setup
```sql
-- Run database_migration.sql to add cloud_training column
```

### 2. Create Google Colab Notebook
1. Go to https://colab.research.google.com
2. Create new notebook
3. **Use `AgriShield_Training_Colab_AUTO.py`** for automatic detection (recommended!)
   - OR use `AgriShield_Training_Colab.py` for manual job selection
4. Copy sections into separate cells
5. **IMPORTANT:** Run CELL 5 (auto-polling) and **KEEP IT RUNNING**
   - This cell automatically detects new training jobs
   - You can minimize the tab - it works in background
6. Save to Google Drive
7. Get shareable link and copy notebook ID

### 3. Update Configuration
- Edit `cloud_training_helper.php`
- Replace `YOUR_NOTEBOOK_ID_HERE` with your Colab notebook ID

### 4. Upload Dataset
- Upload dataset to Google Drive: `/content/drive/MyDrive/AgriShield/datasets/`
- Or create API endpoint to download from server

### 5. Add Training Code
- Copy training logic from `admin_training_script.py`
- Paste into CELL 8 of Colab notebook

### 6. Test
- **Make sure CELL 5 is running in Colab** (auto-polling active)
- Go to admin interface → Training tab
- Select "Cloud Training (Google Colab)"
- Click "Start Training"
- **Training starts automatically!** (Colab detects new job within 30 seconds)
- No need to open Colab or click anything - it's fully automatic!

---

## 🔄 How It Works (AUTOMATIC!)

1. **One-Time Setup:**
   - Create Colab notebook once
   - Run CELL 5 (auto-polling) and **keep it running**
   - Minimize the tab - it works in background

2. **Admin clicks "Train in Cloud"** ✅ AUTOMATIC
   - PHP creates training job in database with `cloud_training = 1`
   - Status set to `pending`

3. **Colab auto-detects new job** ✅ AUTOMATIC
   - Auto-polling cell checks database every 30 seconds
   - Detects new pending job automatically
   - Gets training parameters (epochs, batch_size)
   - Updates status to `running`

4. **Training runs in Colab** ✅ AUTOMATIC
   - Uses free GPU
   - Logs progress to database (`training_logs` table)
   - Saves model to Google Drive

5. **Training completes** ✅ AUTOMATIC
   - Status updated to `completed`
   - Model saved to: `/content/drive/MyDrive/AgriShield/models/job_X_model.pth`

6. **Admin downloads model** ⚠️ MANUAL (one-time)
   - Download from Google Drive
   - Upload via admin interface
   - Deploy model

**You DON'T need to open Colab every time!** Just keep CELL 5 running in the background.

---

## 📊 Database Schema

### `training_jobs` table
- Added: `cloud_training` (TINYINT) - 1 for cloud, 0 for local

### `training_logs` table (new)
- `log_id` - Primary key
- `training_job_id` - Foreign key to training_jobs
- `log_level` - INFO, WARNING, ERROR
- `message` - Log message
- `created_at` - Timestamp

---

## 🔧 Configuration

### Database Credentials
Update in `AgriShield_Training_Colab.py` (CELL 3):
```python
DB_CONFIG = {
    'host': 'your_host',
    'user': 'your_user',
    'password': 'your_password',
    'database': 'your_database',
    'charset': 'utf8mb4'
}
```

### Google Colab Notebook ID
Update in `cloud_training_helper.php`:
```php
$colab_notebook_id = "YOUR_NOTEBOOK_ID_HERE";
```

### Dataset Path
Update in `AgriShield_Training_Colab.py` (CELL 7):
```python
DATASET_PATH = '/content/drive/MyDrive/AgriShield/datasets'
```

---

## ✅ Benefits

- ✅ **Free GPU** training (no cost)
- ✅ **No server load** (training runs in cloud)
- ✅ **Integrated** with admin interface
- ✅ **Works from anywhere** (just need internet)
- ✅ **Automatic logging** to database
- ✅ **Easy model download** from Google Drive

---

## 📝 Notes

- Colab sessions timeout after ~90 minutes of inactivity
- Free tier has usage limits (check Google Colab quotas)
- Models saved to Google Drive (15GB free storage)
- Database must be accessible from Google Colab IPs

---

## 🆘 Troubleshooting

### "No pending cloud training job found"
- Make sure you clicked "Start Training" from admin interface
- Check that `cloud_training = 1` in database

### "Dataset not found"
- Upload dataset to Google Drive
- Update `DATASET_PATH` in Colab notebook

### "Database connection failed"
- Check database credentials
- Ensure database allows connections from Google Colab IPs
- Test connection from Colab: `pymysql.connect(**DB_CONFIG)`

### "Training code not working"
- Copy full training logic from `admin_training_script.py`
- Replace placeholder in CELL 8
- Check for import errors

---

## 📚 Next Steps

1. Complete setup steps above
2. Test with small dataset first
3. Monitor training logs in admin interface
4. Download and deploy trained models

---

**Ready to train in the cloud!** ☁️🚀

