# Setup Heroku Training Service

## 🎯 Fully Independent Training on Heroku

Deploy training service to Heroku - runs automatically, no browser needed!

---

## 🚀 Quick Setup

### Step 1: Create Heroku App

```bash
# Install Heroku CLI if not installed
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create new app
cd colab_training/heroku_training_service
heroku create your-training-service-name

# Or create via Heroku dashboard:
# https://dashboard.heroku.com/new
```

### Step 2: Set Environment Variables

```bash
# Set database credentials
heroku config:set DB_HOST=auth-db1322.hstgr.io
heroku config:set DB_USER=u520834156_uAShield2025
heroku config:set DB_PASSWORD=:JqjB0@0zb6v
heroku config:set DB_NAME=u520834156_dbAgriShield
heroku config:set TRAINING_SCRIPT=train.py
```

Or set via Heroku dashboard:
1. Go to your app → Settings
2. Click "Reveal Config Vars"
3. Add each variable

### Step 3: Deploy

**Option A: Git Deploy (Recommended)**
```bash
# Initialize git if not already
git init
git add .
git commit -m "Initial commit"

# Add Heroku remote
heroku git:remote -a your-training-service-name

# Deploy
git push heroku main
```

**Option B: GitHub Deploy**
1. Push code to GitHub
2. Go to Heroku dashboard → Deploy
3. Connect GitHub repository
4. Enable automatic deploys

### Step 4: Get Service URL

After deployment, Heroku provides URL:
```
https://your-training-service-name.herokuapp.com
```

### Step 5: Update PHP Backend

Edit `colab_training/cloud_training_helper.php`:

```php
$training_service_url = "https://your-training-service-name.herokuapp.com";
```

Or set environment variable:
```bash
# In your PHP server
export TRAINING_SERVICE_URL=https://your-training-service-name.herokuapp.com
```

### Step 6: Test

1. Go to admin interface
2. Click "Train in Cloud"
3. Training starts automatically!
4. Check status in admin interface

---

## 📝 Important Notes

### Heroku Slug Size Limit

Heroku has a **500MB slug size limit**. PyTorch is large (~450MB), so:

**Option 1: Use CPU-only PyTorch** (recommended)
- Smaller size (~150MB)
- Fits Heroku limit
- Slower training (no GPU)

**Option 2: Use ONNX Runtime for inference**
- Very small (~40MB)
- Good for inference
- Not for training

**Option 3: Use Heroku with GPU** (paid)
- Heroku doesn't offer GPU by default
- Consider Railway/Render for GPU

### Training Script

You need to upload your training script to Heroku:

1. Copy `admin_training_script.py` to `heroku_training_service/train.py`
2. Or set `TRAINING_SCRIPT` environment variable to path
3. Commit and deploy

### Timeout Limits

- Heroku free tier: 30 seconds request timeout
- Heroku paid: 30 seconds request timeout
- **Solution**: Training runs in background thread, returns immediately

### Memory Limits

- Heroku free tier: 512MB RAM
- Heroku paid: 512MB-14GB RAM
- PyTorch training may need more memory

---

## 🔧 Troubleshooting

### "Application Error"
- Check Heroku logs: `heroku logs --tail`
- Verify environment variables are set
- Check database connection

### "Training script not found"
- Upload training script to Heroku
- Set `TRAINING_SCRIPT` environment variable
- Check file paths

### "Slug size too large"
- Use CPU-only PyTorch
- Remove unused dependencies
- Use ONNX Runtime if possible

### "Training timeout"
- Training runs in background (no timeout)
- Check Heroku logs for errors
- Verify database connection

---

## ✅ Benefits

- ✅ **Fully automatic** - no browser needed
- ✅ **Runs 24/7** - always available
- ✅ **Independent** - no manual steps
- ✅ **Free tier** available (with limitations)
- ✅ **Easy deployment** - same as your ML API

---

## 💰 Cost

- **Free tier**: 550-1000 hours/month
- **Hobby**: $7/month (no sleep, more resources)
- **Standard**: $25/month (better performance)

---

## 🚀 Next Steps

1. Deploy to Heroku (follow steps above)
2. Update PHP with Heroku URL
3. Test training
4. Monitor logs: `heroku logs --tail`

**Training runs automatically - no browser needed!** 🎯

