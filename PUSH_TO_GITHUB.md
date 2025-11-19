# 🚀 Push to GitHub - Instructions

## 📋 **Steps to Push Flask Folder to GitHub:**

### **Step 1: Navigate to Flask Folder**
```bash
cd AgriShield_ML_Flask
```

### **Step 2: Initialize Git (if not already)**
```bash
git init
```

### **Step 3: Add Remote Repository**
```bash
git remote add origin https://github.com/xCaliyPsO/AgriShield_Flask.git
```

### **Step 4: Add All Files**
```bash
git add .
```

### **Step 5: Commit**
```bash
git commit -m "Initial commit: AgriShield Flask ML API with 3 ML systems"
```

### **Step 6: Push to GitHub**
```bash
git branch -M main
git push -u origin main
```

---

## 🔐 **If Authentication Required:**

### **Option 1: Personal Access Token**
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` permissions
3. Use token as password when pushing

### **Option 2: SSH Key**
```bash
# Change remote to SSH
git remote set-url origin git@github.com:xCaliyPsO/AgriShield_Flask.git
git push -u origin main
```

---

## ✅ **After Pushing:**

Your repository will have:
- ✅ All Flask application files
- ✅ Configuration files
- ✅ Startup scripts
- ✅ Documentation
- ✅ Requirements file

**Repository URL:** https://github.com/xCaliyPsO/AgriShield_Flask.git

---

## 📝 **Note:**

- Model files (`*.pt`) are excluded via `.gitignore` (too large)
- Environment files (`.env`) are excluded for security
- Log files are excluded

