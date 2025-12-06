# Epochs Configuration Guide

## 🎯 Setting Default Epochs System-Wide

You can now configure the default epochs and batch size for your entire system using environment variables.

## 📋 Environment Variables

Set these environment variables to configure training defaults:

- **`DEFAULT_EPOCHS`** - Default number of training epochs (default: 50)
- **`DEFAULT_BATCH_SIZE`** - Default training batch size (default: 8)

## 🚀 How to Set

### **Option 1: Environment Variables (Recommended)**

#### **Windows (PowerShell):**
```powershell
$env:DEFAULT_EPOCHS = "100"
$env:DEFAULT_BATCH_SIZE = "16"
```

#### **Windows (Command Prompt):**
```cmd
set DEFAULT_EPOCHS=100
set DEFAULT_BATCH_SIZE=16
```

#### **Linux/Mac:**
```bash
export DEFAULT_EPOCHS=100
export DEFAULT_BATCH_SIZE=16
```

#### **Heroku:**
```bash
heroku config:set DEFAULT_EPOCHS=100
heroku config:set DEFAULT_BATCH_SIZE=16
```

### **Option 2: .env File (Local Development)**

Create a `.env` file in your project root:
```
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=16
```

Then load it before running:
```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { $line = $_ -split '='; [Environment]::SetEnvironmentVariable($line[0], $line[1], "Process") }
```

### **Option 3: System Environment Variables (Permanent)**

#### **Windows:**
1. Open System Properties → Environment Variables
2. Add new variable:
   - Name: `DEFAULT_EPOCHS`
   - Value: `100`
3. Restart your terminal/IDE

#### **Linux/Mac:**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export DEFAULT_EPOCHS=100
export DEFAULT_BATCH_SIZE=16
```

## 📍 Where It's Used

The environment variables are used in:

1. **`pest_detection_api.py`** - Main API service
   - Lines: Default epochs/batch_size configuration
   - Used when: Training job doesn't specify epochs/batch_size

2. **`training/heroku_training_service/app.py`** - Training service
   - Lines: Default epochs/batch_size configuration
   - Used when: Training job doesn't specify epochs/batch_size

3. **`training/heroku_training_service/train.py`** - Training script
   - Lines: Argument parser defaults
   - Used when: Training script is called without --epochs parameter

## ✅ Verification

After setting the environment variables, restart your application and check the logs. You should see:

```
📊 Training defaults configured:
   Default epochs: 100
   Default batch size: 16
```

## 🔄 Priority Order

The system uses values in this priority order:

1. **Job-specific value** (from database `epochs` column or `training_config` JSON)
2. **Environment variable** (`DEFAULT_EPOCHS` / `DEFAULT_BATCH_SIZE`)
3. **Hardcoded fallback** (50 for epochs, 8 for batch_size)

## 📝 Examples

### Example 1: Set epochs to 100 for all training
```bash
export DEFAULT_EPOCHS=100
# Now all training jobs will use 100 epochs unless specified otherwise
```

### Example 2: Set batch size to 16
```bash
export DEFAULT_BATCH_SIZE=16
# Now all training jobs will use batch size 16 unless specified otherwise
```

### Example 3: Override for specific job
Even with environment variables set, you can still override for specific jobs:
- Set `epochs` column in database
- Or set `epochs` in `training_config` JSON

## ⚠️ Notes

- Environment variables must be set **before** starting the application
- Changes require application restart to take effect
- Job-specific values always take priority over environment variables
- If environment variable is not set, defaults to 50 (epochs) and 8 (batch_size)

