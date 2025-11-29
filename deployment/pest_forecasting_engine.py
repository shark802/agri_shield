#!/usr/bin/env python3
"""
AgriShield Pest Forecasting Engine
Predicts pest outbreaks using weather data and historical pest detection data
"""

import pandas as pd
import numpy as np
import json
import pymysql
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PestForecastingEngine:
    def __init__(self):
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'asdb',
            'charset': 'utf8mb4'
        }
        
        # Pest types from your system
        self.pest_types = [
            'rice_bug',
            'green_leaf_hopper', 
            'black_bug',
            'brown_plant_hopper'
        ]
        
        # Weather factors that affect pest activity
        self.weather_factors = [
            'temperature',
            'humidity',
            'wind_speed',
            'rainfall_1h',
            'pressure',
            'cloudiness'
        ]
        
        # Models for each pest type
        self.models = {}
        self.scalers = {}
        
        # Pest activity thresholds (based on agricultural research)
        self.pest_thresholds = {
            'rice_bug': {
                'low': {'temp_min': 20, 'temp_max': 30, 'humidity_min': 60},
                'medium': {'temp_min': 25, 'temp_max': 35, 'humidity_min': 70},
                'high': {'temp_min': 28, 'temp_max': 38, 'humidity_min': 80}
            },
            'green_leaf_hopper': {
                'low': {'temp_min': 18, 'temp_max': 28, 'humidity_min': 65},
                'medium': {'temp_min': 22, 'temp_max': 32, 'humidity_min': 75},
                'high': {'temp_min': 25, 'temp_max': 35, 'humidity_min': 85}
            },
            'black_bug': {
                'low': {'temp_min': 22, 'temp_max': 30, 'humidity_min': 70},
                'medium': {'temp_min': 25, 'temp_max': 33, 'humidity_min': 80},
                'high': {'temp_min': 28, 'temp_max': 36, 'humidity_min': 90}
            },
            'brown_plant_hopper': {
                'low': {'temp_min': 20, 'temp_max': 28, 'humidity_min': 65},
                'medium': {'temp_min': 24, 'temp_max': 32, 'humidity_min': 75},
                'high': {'temp_min': 27, 'temp_max': 35, 'humidity_min': 85}
            }
        }

    def get_historical_pest_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical pest detection data"""
        try:
            connection = pymysql.connect(**self.db_config)
            
            query = """
            SELECT 
                DATE(created_at) as date,
                classification_json,
                device_id,
                parcel_ref
            FROM images_inbox 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND classification_json IS NOT NULL 
            AND classification_json != ''
            ORDER BY created_at DESC
            """
            
            df = pd.read_sql(query, connection, params=[days_back])
            connection.close()
            
            # Parse pest counts from classification_json
            pest_data = []
            for _, row in df.iterrows():
                try:
                    classification = json.loads(row['classification_json'])
                    if 'pest_counts' in classification:
                        pest_counts = classification['pest_counts']
                        
                        # Calculate total pests detected
                        total_pests = sum(pest_counts.values())
                        
                        # Add individual pest counts
                        pest_record = {
                            'date': row['date'],
                            'total_pests': total_pests,
                            'device_id': row['device_id'],
                            'parcel_ref': row['parcel_ref']
                        }
                        
                        for pest_type in self.pest_types:
                            pest_record[f'{pest_type}_count'] = pest_counts.get(pest_type, 0)
                        
                        pest_data.append(pest_record)
                except json.JSONDecodeError:
                    continue
            
            if pest_data:
                return pd.DataFrame(pest_data)
            else:
                # Return empty DataFrame with correct columns
                columns = ['date', 'total_pests', 'device_id', 'parcel_ref'] + [f'{pest}_count' for pest in self.pest_types]
                return pd.DataFrame(columns=columns)
                
        except Exception as e:
            logger.error(f"Error getting historical pest data: {e}")
            return pd.DataFrame()

    def get_historical_weather_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical weather data"""
        try:
            connection = pymysql.connect(**self.db_config)
            
            query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(temperature) as avg_temperature,
                AVG(humidity) as avg_humidity,
                AVG(wind_speed) as avg_wind_speed,
                SUM(rainfall_1h) as total_rainfall,
                AVG(pressure) as avg_pressure,
                AVG(cloudiness) as avg_cloudiness,
                MAX(temperature) as max_temperature,
                MIN(temperature) as min_temperature
            FROM weather_current 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            """
            
            df = pd.read_sql(query, connection, params=[days_back])
            connection.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical weather data: {e}")
            return pd.DataFrame()

    def get_weather_forecast(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get weather forecast data"""
        try:
            connection = pymysql.connect(**self.db_config)
            
            query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(temperature) as avg_temperature,
                AVG(humidity) as avg_humidity,
                AVG(wind_speed) as avg_wind_speed,
                SUM(rainfall_3h) as total_rainfall,
                AVG(pressure) as avg_pressure,
                AVG(cloudiness) as avg_cloudiness
            FROM weather_forecast 
            WHERE timestamp >= NOW() 
            AND timestamp <= DATE_ADD(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
            """
            
            df = pd.read_sql(query, connection, params=[days_ahead])
            connection.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return pd.DataFrame()

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data by combining pest and weather data"""
        logger.info("Preparing training data...")
        
        # Get historical data
        pest_data = self.get_historical_pest_data(30)
        weather_data = self.get_historical_weather_data(30)
        
        if pest_data.empty or weather_data.empty:
            logger.warning("Insufficient historical data for training")
            return pd.DataFrame(), pd.DataFrame()
        
        # Merge pest and weather data by date
        combined_data = pd.merge(pest_data, weather_data, on='date', how='inner')
        
        # Create features for machine learning
        features = []
        targets = []
        
        for _, row in combined_data.iterrows():
            # Weather features
            feature_vector = [
                row['avg_temperature'],
                row['avg_humidity'], 
                row['avg_wind_speed'],
                row['total_rainfall'],
                row['avg_pressure'],
                row['avg_cloudiness'],
                row['max_temperature'],
                row['min_temperature']
            ]
            
            features.append(feature_vector)
            
            # Pest targets (total pests detected)
            targets.append(row['total_pests'])
        
        X = pd.DataFrame(features, columns=[
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloudiness', 'max_temp', 'min_temp'
        ])
        
        y = pd.Series(targets, name='total_pests')
        
        logger.info(f"Prepared {len(X)} training samples")
        return X, y

    def train_models(self):
        """Train forecasting models"""
        logger.info("Training pest forecasting models...")
        
        X, y = self.prepare_training_data()
        
        if X.empty:
            logger.warning("No training data available, using rule-based forecasting")
            return
        
        # Train models for different pest types
        for pest_type in self.pest_types:
            try:
                # Get pest-specific targets
                pest_targets = []
                for _, row in X.iterrows():
                    # This is a simplified approach - in practice you'd have pest-specific targets
                    pest_targets.append(y.iloc[len(pest_targets)])
                
                if len(pest_targets) < 5:  # Need minimum data for training
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train Random Forest model
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42,
                    min_samples_split=2
                )
                
                model.fit(X_scaled, pest_targets)
                
                self.models[pest_type] = model
                self.scalers[pest_type] = scaler
                
                logger.info(f"Trained model for {pest_type}")
                
            except Exception as e:
                logger.error(f"Error training model for {pest_type}: {e}")

    def predict_pest_risk(self, weather_data: Dict) -> Dict:
        """Predict pest risk based on weather conditions"""
        risk_predictions = {}
        
        for pest_type in self.pest_types:
            try:
                # Rule-based risk assessment
                temp = weather_data.get('temperature', 25)
                humidity = weather_data.get('humidity', 70)
                
                risk_level = 'low'
                risk_score = 0.3
                
                # Check against pest-specific thresholds
                thresholds = self.pest_thresholds.get(pest_type, {})
                
                if 'high' in thresholds:
                    high_thresh = thresholds['high']
                    if (temp >= high_thresh['temp_min'] and 
                        temp <= high_thresh['temp_max'] and 
                        humidity >= high_thresh['humidity_min']):
                        risk_level = 'high'
                        risk_score = 0.9
                    elif 'medium' in thresholds:
                        med_thresh = thresholds['medium']
                        if (temp >= med_thresh['temp_min'] and 
                            temp <= med_thresh['temp_max'] and 
                            humidity >= med_thresh['humidity_min']):
                            risk_level = 'medium'
                            risk_score = 0.6
                
                # Additional factors
                if weather_data.get('rainfall', 0) > 5:
                    risk_score += 0.1  # Rain increases pest activity
                
                if weather_data.get('wind_speed', 0) > 10:
                    risk_score -= 0.1  # High wind reduces pest activity
                
                risk_score = max(0.1, min(1.0, risk_score))
                
                risk_predictions[pest_type] = {
                    'risk_level': risk_level,
                    'risk_score': round(risk_score, 2),
                    'confidence': 0.8
                }
                
            except Exception as e:
                logger.error(f"Error predicting risk for {pest_type}: {e}")
                risk_predictions[pest_type] = {
                    'risk_level': 'unknown',
                    'risk_score': 0.5,
                    'confidence': 0.3
                }
        
        return risk_predictions

    def generate_forecast(self, days_ahead: int = 7) -> Dict:
        """Generate pest forecast for next N days"""
        logger.info(f"Generating {days_ahead}-day pest forecast...")
        
        # Get weather forecast
        weather_forecast = self.get_weather_forecast(days_ahead)
        
        if weather_forecast.empty:
            logger.warning("No weather forecast available")
            return {'error': 'No weather forecast data available'}
        
        forecast_results = {
            'generated_at': datetime.now().isoformat(),
            'forecast_days': days_ahead,
            'daily_forecasts': []
        }
        
        for _, weather_day in weather_forecast.iterrows():
            weather_data = {
                'temperature': weather_day['avg_temperature'],
                'humidity': weather_day['avg_humidity'],
                'wind_speed': weather_day['avg_wind_speed'],
                'rainfall': weather_day['total_rainfall'],
                'pressure': weather_day['avg_pressure'],
                'cloudiness': weather_day['avg_cloudiness']
            }
            
            # Predict pest risk for this day
            pest_risks = self.predict_pest_risk(weather_data)
            
            # Calculate overall risk
            overall_risk_scores = [risk['risk_score'] for risk in pest_risks.values()]
            overall_risk = np.mean(overall_risk_scores) if overall_risk_scores else 0.5
            
            if overall_risk >= 0.7:
                overall_level = 'high'
            elif overall_risk >= 0.4:
                overall_level = 'medium'
            else:
                overall_level = 'low'
            
            day_forecast = {
                'date': weather_day['date'].strftime('%Y-%m-%d'),
                'weather': weather_data,
                'overall_risk': {
                    'level': overall_level,
                    'score': round(overall_risk, 2)
                },
                'pest_risks': pest_risks,
                'recommendations': self.generate_recommendations(pest_risks, weather_data)
            }
            
            forecast_results['daily_forecasts'].append(day_forecast)
        
        return forecast_results

    def generate_recommendations(self, pest_risks: Dict, weather_data: Dict) -> List[str]:
        """Generate recommendations based on pest risks and weather"""
        recommendations = []
        
        high_risk_pests = [pest for pest, risk in pest_risks.items() 
                          if risk['risk_level'] == 'high']
        
        if high_risk_pests:
            recommendations.append(f"High risk detected for: {', '.join(high_risk_pests)}")
            recommendations.append("Consider preventive treatment with recommended pesticides")
            recommendations.append("Monitor fields closely for pest activity")
        
        if weather_data.get('humidity', 0) > 80:
            recommendations.append("High humidity detected - favorable for pest reproduction")
            recommendations.append("Ensure proper field drainage")
        
        if weather_data.get('temperature', 0) > 32:
            recommendations.append("High temperature detected - monitor for heat stress")
            recommendations.append("Consider irrigation to reduce plant stress")
        
        if weather_data.get('rainfall', 0) > 10:
            recommendations.append("Heavy rainfall expected - check for standing water")
            recommendations.append("Pests may seek refuge in plants")
        
        if not recommendations:
            recommendations.append("Weather conditions are favorable for pest control")
            recommendations.append("Continue regular monitoring")
        
        return recommendations

    def save_forecast_to_database(self, forecast_data: Dict):
        """Save forecast results to database"""
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Create forecast table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS pest_forecasts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                forecast_date DATE NOT NULL,
                pest_type VARCHAR(50) NOT NULL,
                risk_level ENUM('low', 'medium', 'high') NOT NULL,
                risk_score DECIMAL(3,2) NOT NULL,
                confidence DECIMAL(3,2) NOT NULL,
                weather_temperature DECIMAL(5,2),
                weather_humidity DECIMAL(5,2),
                weather_rainfall DECIMAL(5,2),
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_forecast_date (forecast_date),
                INDEX idx_pest_type (pest_type)
            )
            """
            
            cursor.execute(create_table_query)
            
            # Insert forecast data
            for day_forecast in forecast_data.get('daily_forecasts', []):
                forecast_date = day_forecast['date']
                weather = day_forecast['weather']
                
                for pest_type, risk_info in day_forecast['pest_risks'].items():
                    insert_query = """
                    INSERT INTO pest_forecasts 
                    (forecast_date, pest_type, risk_level, risk_score, confidence,
                     weather_temperature, weather_humidity, weather_rainfall, recommendations)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(insert_query, (
                        forecast_date,
                        pest_type,
                        risk_info['risk_level'],
                        risk_info['risk_score'],
                        risk_info['confidence'],
                        weather['temperature'],
                        weather['humidity'],
                        weather['rainfall'],
                        json.dumps(day_forecast['recommendations'])
                    ))
            
            connection.commit()
            connection.close()
            
            logger.info("Forecast data saved to database")
            
        except Exception as e:
            logger.error(f"Error saving forecast to database: {e}")

def main():
    """Main function to generate pest forecast"""
    print("🌾 AgriShield Pest Forecasting Engine")
    print("=" * 50)
    
    # Initialize forecasting engine
    engine = PestForecastingEngine()
    
    # Train models (if sufficient data available)
    engine.train_models()
    
    # Generate 7-day forecast
    print("🔮 Generating 7-day pest forecast...")
    forecast = engine.generate_forecast(7)
    
    if 'error' in forecast:
        print(f"❌ Error: {forecast['error']}")
        return
    
    # Display results
    print(f"\n📊 Pest Forecast Generated: {forecast['generated_at']}")
    print(f"📅 Forecast Period: {forecast['forecast_days']} days")
    
    for day in forecast['daily_forecasts']:
        print(f"\n📅 {day['date']}:")
        print(f"   🌡️ Temperature: {day['weather']['temperature']:.1f}°C")
        print(f"   💧 Humidity: {day['weather']['humidity']:.1f}%")
        print(f"   🌧️ Rainfall: {day['weather']['rainfall']:.1f}mm")
        print(f"   ⚠️ Overall Risk: {day['overall_risk']['level'].upper()} ({day['overall_risk']['score']})")
        
        print(f"   🐛 Pest Risks:")
        for pest, risk in day['pest_risks'].items():
            print(f"      - {pest}: {risk['risk_level']} ({risk['risk_score']})")
        
        print(f"   💡 Recommendations:")
        for rec in day['recommendations'][:3]:  # Show first 3 recommendations
            print(f"      • {rec}")
    
    # Save to database
    engine.save_forecast_to_database(forecast)
    
    print(f"\n✅ Forecast complete! Data saved to database.")
    print(f"📊 Use the API endpoints to access forecast data.")

if __name__ == "__main__":
    main()





