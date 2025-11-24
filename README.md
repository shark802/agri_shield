# AgriShield Django ML API

Django version of the AgriShield ML Flask API.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

3. **Run development server:**
   ```bash
   python manage.py runserver
   ```

4. **Test:**
   ```bash
   curl http://localhost:8000/health/
   ```

## 📡 Endpoints

- `GET /` - API information
- `GET /health/` - Health check
- `GET /status/` - Status check
- `POST /detect/` - Detect pests (multipart form-data)
- `POST /classify/` - Classify pests (Android app)
- `GET /forecast/` - Generate 7-day pest forecast
- `GET /forecast/quick/` - Quick single forecast
- `GET /forecast/current/` - Get forecast from database
- `POST /forecast/update/` - Manually update forecast

## 🔧 Configuration

Uses the same `config.py` file as the Flask version for database and ML settings.

## 📝 Notes

- Converted from Flask to Django
- Same ML functionality as Flask version
- Uses Django REST framework patterns
- CORS enabled for all origins

