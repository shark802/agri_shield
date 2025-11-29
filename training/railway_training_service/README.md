# Independent Training Service (Railway/Render)

## 🎯 Fully Automatic Training - No Browser Needed!

This service runs training **completely independently** - no Colab, no browser, no manual steps!

---

## 🚀 Deploy to Railway (Recommended)

### Step 1: Create Railway Account
1. Go to https://railway.app
2. Sign up with GitHub
3. Create new project

### Step 2: Deploy Service
1. Connect GitHub repository
2. Railway auto-detects Python
3. Deploys automatically

### Step 3: Set Environment Variables
In Railway dashboard, add:
```
DB_HOST=auth-db1322.hstgr.io
DB_USER=u520834156_uAShield2025
DB_PASSWORD=:JqjB0@0zb6v
DB_NAME=u520834156_dbAgriShield
TRAINING_SCRIPT=/app/train.py
PORT=5000
```

### Step 4: Get Service URL
Railway provides URL like: `https://your-service.railway.app`

---

## 🚀 Deploy to Render (Alternative)

### Step 1: Create Render Account
1. Go to https://render.com
2. Sign up
3. Create new Web Service

### Step 2: Deploy
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`

### Step 3: Set Environment Variables
Same as Railway above

---

## 🔧 Update PHP Backend

Update `cloud_training_helper.php` to use Railway/Render service:

```php
// Instead of Colab URL, use Railway service
$training_service_url = "https://your-service.railway.app";

// Call training service API
$response = file_get_contents($training_service_url . '/train', false, stream_context_create([
    'http' => [
        'method' => 'POST',
        'header' => 'Content-Type: application/json',
        'content' => json_encode(['job_id' => $job_id])
    ]
]));
```

---

## ✅ Benefits

- ✅ **Fully automatic** - no browser needed
- ✅ **Runs 24/7** - always available
- ✅ **Independent** - no manual steps
- ✅ **Free tier** available (Railway/Render)
- ✅ **Auto-scales** - handles multiple jobs

---

## 📝 How It Works

1. Admin clicks "Train in Cloud"
2. PHP creates training job in database
3. PHP calls Railway service API: `POST /train`
4. Service starts training in background
5. Training runs independently
6. Model saved to database/server
7. Status updated automatically

**No browser, no tabs, fully independent!** 🎯

