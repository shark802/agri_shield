"""
Pest Forecasting System for Django
Converted from Flask app.py SimplePestForecaster class
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests

logger = logging.getLogger(__name__)

# Try to import database connection
try:
    import pymysql
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("pymysql not available - forecasting will use provided weather data only")

# Try to load configuration
try:
    from config import (
        DB_CONFIG, DEFAULT_LOCATION,
        PEST_THRESHOLDS,
        RISK_BASE_SCORE, RISK_TEMP_OPTIMAL, RISK_TEMP_NEAR,
        RISK_HUMIDITY_OPTIMAL, RISK_HUMIDITY_NEAR,
        RISK_RAINFALL_MODERATE, RISK_RAINFALL_HEAVY, RISK_WIND_HIGH,
        RISK_RECENT_DETECTION_BOOST, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM,
        FORECAST_DAYS_BACK, FORECAST_MAX_DETECTIONS,
        DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY, DEFAULT_RAINFALL, DEFAULT_WIND_SPEED,
        RAINFALL_MODERATE_MIN, RAINFALL_MODERATE_MAX, RAINFALL_HEAVY_THRESHOLD,
        WIND_HIGH_THRESHOLD, TEMP_NEAR_OPTIMAL_RANGE, HUMIDITY_NEAR_OPTIMAL_RANGE,
        FORECAST_CONFIDENCE, DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
        WEATHERAPI_KEY, WEATHERAPI_BASE_URL
    )
    USE_CONFIG_FILE = True
except ImportError:
    USE_CONFIG_FILE = False
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'asdb'),
        'charset': os.getenv('DB_CHARSET', 'utf8mb4')
    }
    DEFAULT_LOCATION = os.getenv('DEFAULT_LOCATION', 'Bago City')
    PEST_THRESHOLDS = {
        'rice_bug': {'optimal_temp': (28, 32), 'optimal_humidity': (70, 85)},
        'black-bug': {'optimal_temp': (25, 30), 'optimal_humidity': (75, 90)},
        'brown_hopper': {'optimal_temp': (26, 32), 'optimal_humidity': (80, 95)},
        'green_hopper': {'optimal_temp': (25, 30), 'optimal_humidity': (70, 85)},
    }
    RISK_BASE_SCORE = 0.2
    RISK_TEMP_OPTIMAL = 0.4
    RISK_TEMP_NEAR = 0.2
    RISK_HUMIDITY_OPTIMAL = 0.4
    RISK_HUMIDITY_NEAR = 0.2
    RISK_RAINFALL_MODERATE = 0.1
    RISK_RAINFALL_HEAVY = -0.1
    RISK_WIND_HIGH = -0.1
    RISK_RECENT_DETECTION_BOOST = 0.2
    RISK_THRESHOLD_HIGH = 0.7
    RISK_THRESHOLD_MEDIUM = 0.4
    FORECAST_DAYS_BACK = 7
    FORECAST_MAX_DETECTIONS = 20
    DEFAULT_TEMPERATURE = 25
    DEFAULT_HUMIDITY = 70
    DEFAULT_RAINFALL = 0
    DEFAULT_WIND_SPEED = 5
    RAINFALL_MODERATE_MIN = 5
    RAINFALL_MODERATE_MAX = 20
    RAINFALL_HEAVY_THRESHOLD = 20
    WIND_HIGH_THRESHOLD = 15
    TEMP_NEAR_OPTIMAL_RANGE = 3
    HUMIDITY_NEAR_OPTIMAL_RANGE = 10
    FORECAST_CONFIDENCE = 0.7
    DEFAULT_LATITUDE = 10.5388
    DEFAULT_LONGITUDE = 122.8383
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', '76f91b260dc84341a1733851250710')
    WEATHERAPI_BASE_URL = os.getenv('WEATHERAPI_BASE_URL', 'http://api.weatherapi.com/v1')


class SimplePestForecaster:
    """Simple rule-based pest forecasting (no ML dependencies)"""
    
    def __init__(self, db_config=None):
        self.db_config = db_config or DB_CONFIG
        self.pest_types = [
            'rice_bug',
            'green_leaf_hopper', 
            'black_bug',
            'brown_plant_hopper'
        ]
        self.pest_thresholds = PEST_THRESHOLDS
    
    def get_current_weather(self) -> Dict:
        """Get current weather data from database"""
        if not DB_AVAILABLE:
            return {}
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            query = """
            SELECT 
                temperature,
                humidity,
                wind_speed,
                rainfall_1h,
                pressure,
                cloudiness,
                weather_description,
                location_name,
                last_updated
            FROM weather_current 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            cursor.execute(query)
            weather_data = cursor.fetchone()
            connection.close()
            
            if weather_data:
                return dict(weather_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {}
    
    def get_hourly_weather_forecast(self, days: int = 7) -> List[Dict]:
        """Get hourly weather forecast data for next N days from database or API"""
        if not DB_AVAILABLE:
            return []
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            query = """
            SELECT 
                timestamp,
                temperature,
                humidity,
                wind_speed,
                COALESCE(rainfall_3h, rainfall_1h, 0) as rainfall,
                pressure,
                cloudiness,
                weather_description,
                location_lat,
                location_lon
            FROM weather_forecast 
            WHERE timestamp >= NOW()
            AND timestamp <= DATE_ADD(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (days,))
            forecast_data = cursor.fetchall()
            
            if not forecast_data:
                logger.warning("No forecast data in database. Attempting to fetch from WeatherAPI...")
                forecast_data = self._fetch_forecast_from_api(days)
                
                if not forecast_data:
                    logger.warning("API fetch failed. Using current weather with estimates")
                    current_query = """
                    SELECT 
                        temperature,
                        humidity,
                        wind_speed,
                        rainfall_1h,
                        pressure,
                        cloudiness,
                        weather_description,
                        location_name
                    FROM weather_current 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                    """
                    cursor.execute(current_query)
                    current = cursor.fetchone()
                    
                    if current:
                        forecast_data = []
                        for hour in range(days * 24):
                            hour_data = dict(current)
                            hour_data['timestamp'] = (datetime.now() + timedelta(hours=hour)).strftime('%Y-%m-%d %H:00:00')
                            hour_data['rainfall'] = hour_data.get('rainfall_1h', 0)
                            hour_data['temperature'] = float(hour_data.get('temperature', DEFAULT_TEMPERATURE)) + (hour % 24 - 12) * 0.5
                            forecast_data.append(hour_data)
            
            connection.close()
            
            result = []
            for row in forecast_data:
                result.append(dict(row))
            
            return result
        except Exception as e:
            logger.error(f"Error getting hourly weather forecast: {e}")
            return []
    
    def _fetch_forecast_from_api(self, days: int = 7) -> List[Dict]:
        """Fetch hourly weather forecast from WeatherAPI.com"""
        try:
            api_key = WEATHERAPI_KEY
            base_url = WEATHERAPI_BASE_URL
            lat = DEFAULT_LATITUDE
            lon = DEFAULT_LONGITUDE
            
            url = f"{base_url}/forecast.json"
            params = {
                'key': api_key,
                'q': f"{lat},{lon}",
                'days': min(days, 3),
                'aqi': 'no',
                'alerts': 'no'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_data = []
            
            for day_data in data['forecast']['forecastday']:
                date = day_data['date']
                for hour in day_data['hour']:
                    hour_time = hour['time'].split()[1] if ' ' in hour['time'] else hour['time']
                    forecast_item = {
                        'timestamp': f"{date} {hour_time}",
                        'temperature': hour['temp_c'],
                        'humidity': hour['humidity'],
                        'wind_speed': hour['wind_kph'] / 3.6,
                        'rainfall': hour.get('precip_mm', 0),
                        'pressure': hour['pressure_mb'],
                        'cloudiness': hour['cloud'],
                        'weather_description': hour['condition']['text'],
                        'location_lat': lat,
                        'location_lon': lon
                    }
                    forecast_data.append(forecast_item)
            
            if len(forecast_data) < days * 24:
                last_hour = forecast_data[-1] if forecast_data else None
                if last_hour:
                    for day in range(len(forecast_data) // 24, days):
                        for hour in range(24):
                            extended = dict(last_hour)
                            extended['timestamp'] = (datetime.now() + timedelta(days=day, hours=hour)).strftime('%Y-%m-%d %H:00:00')
                            forecast_data.append(extended)
            
            logger.info(f"✅ Fetched {len(forecast_data)} hourly forecast points from WeatherAPI")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error fetching forecast from API: {e}")
            return []
    
    def save_forecast_to_database(self, forecast_data: List[Dict]) -> bool:
        """Save hourly forecast data to weather_forecast table"""
        if not DB_AVAILABLE or not forecast_data:
            return False
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor()
            
            cursor.execute("""
                DELETE FROM weather_forecast 
                WHERE timestamp < DATE_SUB(NOW(), INTERVAL 1 DAY)
            """)
            
            insert_query = """
                INSERT INTO weather_forecast 
                (timestamp, location_lat, location_lon, temperature, humidity, 
                 pressure, wind_speed, wind_direction, rainfall_3h, cloudiness, 
                 weather_description, weather_main, forecast_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'forecast')
                ON DUPLICATE KEY UPDATE
                    temperature = VALUES(temperature),
                    humidity = VALUES(humidity),
                    pressure = VALUES(pressure),
                    wind_speed = VALUES(wind_speed),
                    rainfall_3h = VALUES(rainfall_3h),
                    cloudiness = VALUES(cloudiness),
                    weather_description = VALUES(weather_description)
            """
            
            saved_count = 0
            for hour_data in forecast_data:
                try:
                    timestamp_str = hour_data.get('timestamp', '')
                    if isinstance(timestamp_str, str):
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    else:
                        timestamp = timestamp_str
                    
                    values = (
                        timestamp,
                        hour_data.get('location_lat', DEFAULT_LATITUDE),
                        hour_data.get('location_lon', DEFAULT_LONGITUDE),
                        hour_data.get('temperature'),
                        hour_data.get('humidity'),
                        hour_data.get('pressure'),
                        hour_data.get('wind_speed'),
                        hour_data.get('wind_direction', 0),
                        hour_data.get('rainfall', hour_data.get('rainfall_3h', 0)),
                        hour_data.get('cloudiness', 0),
                        hour_data.get('weather_description', 'Unknown'),
                        'Forecast'
                    )
                    
                    cursor.execute(insert_query, values)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Error saving forecast hour {hour_data.get('timestamp')}: {e}")
                    continue
            
            connection.commit()
            connection.close()
            
            logger.info(f"✅ Saved {saved_count} hourly forecast records to database")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Error saving forecast to database: {e}")
            return False
    
    def update_forecast_automatically(self) -> bool:
        """Automatically fetch and save forecast data from API"""
        try:
            logger.info("🔄 Auto-updating weather forecast...")
            forecast_data = self._fetch_forecast_from_api(days=3)
            
            if forecast_data:
                if self.save_forecast_to_database(forecast_data):
                    logger.info("✅ Forecast automatically updated successfully")
                    return True
                else:
                    logger.error("❌ Failed to save forecast to database")
                    return False
            else:
                logger.warning("⚠️ No forecast data fetched from API")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in automatic forecast update: {e}")
            return False
    
    def get_recent_pest_data(self, days: int = None) -> List[Dict]:
        """Get recent pest detection data"""
        if days is None:
            days = FORECAST_DAYS_BACK
        if not DB_AVAILABLE:
            return []
        
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            query = """
            SELECT 
                created_at,
                classification_json
            FROM images_inbox 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND classification_json IS NOT NULL 
            AND classification_json != ''
            ORDER BY created_at DESC
            LIMIT %s
            """
            
            cursor.execute(query, (days, FORECAST_MAX_DETECTIONS))
            results = cursor.fetchall()
            connection.close()
            
            pest_data = []
            for row in results:
                try:
                    classification = json.loads(row['classification_json'])
                    if 'pest_counts' in classification:
                        pest_data.append({
                            'date': str(row['created_at']),
                            'pest_counts': classification['pest_counts']
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return pest_data
        except Exception as e:
            logger.error(f"Error getting pest data: {e}")
            return []
    
    def calculate_pest_risk(self, pest_type: str, weather_data: Dict) -> Dict:
        """Calculate pest risk based on weather conditions"""
        if pest_type not in self.pest_thresholds:
            return {
                'risk_level': 'unknown', 
                'risk_score': 0.5, 
                'confidence': FORECAST_CONFIDENCE
            }
        
        thresholds = self.pest_thresholds[pest_type]
        temp = weather_data.get('temperature', DEFAULT_TEMPERATURE)
        humidity = weather_data.get('humidity', DEFAULT_HUMIDITY)
        rainfall = weather_data.get('rainfall_1h', DEFAULT_RAINFALL)
        wind_speed = weather_data.get('wind_speed', DEFAULT_WIND_SPEED)
        
        risk_score = RISK_BASE_SCORE
        
        # Temperature factor
        temp_min, temp_max = thresholds['optimal_temp']
        if temp_min <= temp <= temp_max:
            risk_score += RISK_TEMP_OPTIMAL
        elif abs(temp - (temp_min + temp_max) / 2) <= TEMP_NEAR_OPTIMAL_RANGE:
            risk_score += RISK_TEMP_NEAR
        
        # Humidity factor
        humidity_min, humidity_max = thresholds['optimal_humidity']
        if humidity_min <= humidity <= humidity_max:
            risk_score += RISK_HUMIDITY_OPTIMAL
        elif humidity >= humidity_min - HUMIDITY_NEAR_OPTIMAL_RANGE:
            risk_score += RISK_HUMIDITY_NEAR
        
        # Rainfall factor
        if RAINFALL_MODERATE_MIN < rainfall <= RAINFALL_MODERATE_MAX:
            risk_score += RISK_RAINFALL_MODERATE
        elif rainfall > RAINFALL_HEAVY_THRESHOLD:
            risk_score += RISK_RAINFALL_HEAVY
        
        # Wind factor
        if wind_speed > WIND_HIGH_THRESHOLD:
            risk_score += RISK_WIND_HIGH
        
        risk_score = max(0.1, min(1.0, risk_score))
        
        if risk_score >= RISK_THRESHOLD_HIGH:
            risk_level = 'high'
        elif risk_score >= RISK_THRESHOLD_MEDIUM:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': FORECAST_CONFIDENCE
        }
    
    def generate_forecast(self, weather_data: Dict = None, days: int = 7) -> Dict:
        """Generate comprehensive pest forecast for N days with hourly data"""
        hourly_weather = self.get_hourly_weather_forecast(days)
        
        if not hourly_weather:
            if not weather_data:
                weather_data = self.get_current_weather()
            
            if not weather_data:
                return {'error': 'No weather data available'}
            
            return self._generate_single_forecast(weather_data)
        
        recent_pests = self.get_recent_pest_data(7)
        
        daily_forecasts = {}
        for hour_data in hourly_weather:
            timestamp_str = hour_data.get('timestamp', '')
            try:
                if isinstance(timestamp_str, str):
                    try:
                        hour_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        hour_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                else:
                    hour_dt = timestamp_str
                date_key = hour_dt.strftime('%Y-%m-%d')
            except:
                date_key = datetime.now().strftime('%Y-%m-%d')
            
            if date_key not in daily_forecasts:
                daily_forecasts[date_key] = {
                    'date': date_key,
                    'hours': []
                }
            
            try:
                hour_value = hour_dt.hour if 'hour_dt' in locals() and hour_dt else 0
            except:
                hour_value = 0
            
            daily_forecasts[date_key]['hours'].append({
                'hour': hour_value,
                'timestamp': timestamp_str,
                'temperature': hour_data.get('temperature'),
                'humidity': hour_data.get('humidity'),
                'rainfall': hour_data.get('rainfall', hour_data.get('rainfall_1h', 0)),
                'wind_speed': hour_data.get('wind_speed', 0),
                'weather_description': hour_data.get('weather_description', 'Unknown')
            })
        
        forecast_days = []
        for date_key in sorted(daily_forecasts.keys())[:days]:
            day_data = daily_forecasts[date_key]
            hours = day_data['hours']
            if not hours:
                continue
            
            avg_temp = sum(h.get('temperature', DEFAULT_TEMPERATURE) for h in hours) / len(hours)
            avg_humidity = sum(h.get('humidity', DEFAULT_HUMIDITY) for h in hours) / len(hours)
            total_rainfall = sum(h.get('rainfall', 0) for h in hours)
            avg_wind = sum(h.get('wind_speed', 0) for h in hours) / len(hours)
            
            daily_weather = {
                'temperature': avg_temp,
                'humidity': avg_humidity,
                'rainfall_1h': total_rainfall / len(hours),
                'wind_speed': avg_wind,
                'weather_description': hours[0].get('weather_description', 'Unknown')
            }
            
            pest_risks = {}
            for pest_type in self.pest_types:
                risk = self.calculate_pest_risk(pest_type, daily_weather)
                
                if recent_pests:
                    recent_count = 0
                    for detection in recent_pests:
                        pest_counts = detection.get('pest_counts', {})
                        pest_key = pest_type.replace('_', '-')
                        if pest_key == 'rice-bug':
                            pest_key = 'Rice_Bug'
                        elif pest_key == 'black-bug':
                            pest_key = 'black-bug'
                        elif pest_key == 'brown-plant-hopper':
                            pest_key = 'brown_hopper'
                        elif pest_key == 'green-leaf-hopper':
                            pest_key = 'green_hopper'
                        
                        recent_count += pest_counts.get(pest_key, 0)
                    
                    if recent_count > 0:
                        risk['risk_score'] = min(1.0, risk['risk_score'] + RISK_RECENT_DETECTION_BOOST)
                        if risk['risk_score'] >= RISK_THRESHOLD_HIGH:
                            risk['risk_level'] = 'high'
                        elif risk['risk_score'] >= RISK_THRESHOLD_MEDIUM:
                            risk['risk_level'] = 'medium'
                
                pest_risks[pest_type] = risk
            
            risk_scores = [risk['risk_score'] for risk in pest_risks.values()]
            overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
            
            if overall_risk_score >= RISK_THRESHOLD_HIGH:
                overall_level = 'high'
            elif overall_risk_score >= RISK_THRESHOLD_MEDIUM:
                overall_level = 'medium'
            else:
                overall_level = 'low'
            
            forecast_days.append({
                'date': date_key,
                'overall_risk': {
                    'level': overall_level,
                    'score': round(overall_risk_score, 2)
                },
                'pest_risks': pest_risks,
                'weather': {
                    'temperature_avg': round(avg_temp, 1),
                    'humidity_avg': round(avg_humidity, 1),
                    'rainfall_total': round(total_rainfall, 2),
                    'wind_speed_avg': round(avg_wind, 1),
                    'weather_description': daily_weather['weather_description']
                },
                'hourly_data': hours
            })
        
        all_risk_scores = []
        for day in forecast_days:
            all_risk_scores.append(day['overall_risk']['score'])
        
        overall_7day_risk = sum(all_risk_scores) / len(all_risk_scores) if all_risk_scores else 0.5
        
        if overall_7day_risk >= RISK_THRESHOLD_HIGH:
            overall_7day_level = 'high'
        elif overall_7day_risk >= RISK_THRESHOLD_MEDIUM:
            overall_7day_level = 'medium'
        else:
            overall_7day_level = 'low'
        
        return {
            'generated_at': datetime.now().isoformat(),
            'location': hourly_weather[0].get('location_name', DEFAULT_LOCATION) if hourly_weather else DEFAULT_LOCATION,
            'forecast_period': f'{days} days',
            'overall_7day_risk': {
                'level': overall_7day_level,
                'score': round(overall_7day_risk, 2)
            },
            'daily_forecasts': forecast_days,
            'recent_detections': len(recent_pests)
        }
    
    def _generate_single_forecast(self, weather_data: Dict) -> Dict:
        """Generate single forecast from current weather (fallback method)"""
        recent_pests = self.get_recent_pest_data(7)
        
        pest_risks = {}
        for pest_type in self.pest_types:
            risk = self.calculate_pest_risk(pest_type, weather_data)
            
            if recent_pests:
                recent_count = 0
                for detection in recent_pests:
                    pest_counts = detection.get('pest_counts', {})
                    pest_key = pest_type.replace('_', '-')
                    if pest_key == 'rice-bug':
                        pest_key = 'Rice_Bug'
                    elif pest_key == 'black-bug':
                        pest_key = 'black-bug'
                    elif pest_key == 'brown-plant-hopper':
                        pest_key = 'brown_hopper'
                    elif pest_key == 'green-leaf-hopper':
                        pest_key = 'green_hopper'
                    
                    recent_count += pest_counts.get(pest_key, 0)
                
                if recent_count > 0:
                    risk['risk_score'] = min(1.0, risk['risk_score'] + RISK_RECENT_DETECTION_BOOST)
                    if risk['risk_score'] >= RISK_THRESHOLD_HIGH:
                        risk['risk_level'] = 'high'
                    elif risk['risk_score'] >= RISK_THRESHOLD_MEDIUM:
                        risk['risk_level'] = 'medium'
            
            pest_risks[pest_type] = risk
        
        risk_scores = [risk['risk_score'] for risk in pest_risks.values()]
        overall_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        
        if overall_risk_score >= RISK_THRESHOLD_HIGH:
            overall_level = 'high'
        elif overall_risk_score >= RISK_THRESHOLD_MEDIUM:
            overall_level = 'medium'
        else:
            overall_level = 'low'
        
        return {
            'generated_at': datetime.now().isoformat(),
            'location': weather_data.get('location_name', DEFAULT_LOCATION),
            'current_weather': {
                'temperature': weather_data.get('temperature'),
                'humidity': weather_data.get('humidity'),
                'rainfall': weather_data.get('rainfall_1h', 0),
                'wind_speed': weather_data.get('wind_speed', 0),
                'weather_description': weather_data.get('weather_description', 'Unknown')
            },
            'overall_risk': {
                'level': overall_level,
                'score': round(overall_risk_score, 2)
            },
            'pest_risks': pest_risks,
            'recent_detections': len(recent_pests)
        }


# Initialize forecaster
_forecaster = None

def get_forecaster():
    """Get or create forecaster instance"""
    global _forecaster
    if _forecaster is None:
        _forecaster = SimplePestForecaster()
    return _forecaster

