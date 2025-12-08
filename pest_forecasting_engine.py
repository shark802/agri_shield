#!/usr/bin/env python3
"""
AgriShield Pest Forecasting Engine
Predicts pest outbreaks using weather data and historical pest detection data
"""

import pandas as pd
import numpy as np
import json
import os
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
    def __init__(self, db_config=None):
        """
        Initialize forecasting engine
        
        Args:
            db_config: Optional database config dict. If None, uses environment variables.
        """
        # Database configuration - supports Heroku (env vars) and local development
        if db_config:
            self.db_config = db_config
        else:
            # Use environment variables (for Heroku) or defaults (for local)
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'database': os.getenv('DB_NAME', 'asdb'),
                'charset': os.getenv('DB_CHARSET', 'utf8mb4')
            }
        
        logger.info(f"Forecasting engine initialized with DB: {self.db_config['host']}/{self.db_config['database']}")
        
        # Pest types from your system (matching actual detection data)
        # Note: These match the pest types in pest_class_percentage.php and pest_reports_barangay.php
        self.pest_types = [
            'Rice_Bug',           # Matches detection data
            'green_hopper',       # Matches detection data (green_leaf_hopper)
            'black-bug',          # Matches detection data
            'brown_hopper',       # Matches detection data (brown_plant_hopper)
            'White_stem_borer'    # Matches detection data
        ]
        
        # Pest type mapping (detection format -> standard format)
        self.pest_type_mapping = {
            'rice_bug': 'Rice_Bug',
            'green_leaf_hopper': 'green_hopper',
            'green_hopper': 'green_hopper',
            'black_bug': 'black-bug',
            'black-bug': 'black-bug',
            'brown_plant_hopper': 'brown_hopper',
            'brown_hopper': 'brown_hopper',
            'white_stem_borer': 'White_stem_borer',
            'White_stem_borer': 'White_stem_borer'
        }
        
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
        # Updated to match actual pest type names in detection data
        self.pest_thresholds = {
            'Rice_Bug': {
                'low': {'temp_min': 20, 'temp_max': 30, 'humidity_min': 60},
                'medium': {'temp_min': 25, 'temp_max': 35, 'humidity_min': 70},
                'high': {'temp_min': 28, 'temp_max': 38, 'humidity_min': 80}
            },
            'green_hopper': {
                'low': {'temp_min': 18, 'temp_max': 28, 'humidity_min': 65},
                'medium': {'temp_min': 22, 'temp_max': 32, 'humidity_min': 75},
                'high': {'temp_min': 25, 'temp_max': 35, 'humidity_min': 85}
            },
            'black-bug': {
                'low': {'temp_min': 22, 'temp_max': 30, 'humidity_min': 70},
                'medium': {'temp_min': 25, 'temp_max': 33, 'humidity_min': 80},
                'high': {'temp_min': 28, 'temp_max': 36, 'humidity_min': 90}
            },
            'brown_hopper': {
                'low': {'temp_min': 20, 'temp_max': 28, 'humidity_min': 65},
                'medium': {'temp_min': 24, 'temp_max': 32, 'humidity_min': 75},
                'high': {'temp_min': 27, 'temp_max': 35, 'humidity_min': 85}
            },
            'White_stem_borer': {
                'low': {'temp_min': 22, 'temp_max': 30, 'humidity_min': 70},
                'medium': {'temp_min': 25, 'temp_max': 33, 'humidity_min': 80},
                'high': {'temp_min': 28, 'temp_max': 36, 'humidity_min': 85}
            }
        }

    def get_historical_pest_data(self, days_back: int = 30, farm_id: int = None, barangay: str = None) -> pd.DataFrame:
        """
        Get historical pest detection data (matching pest_class_percentage.php and pest_reports_barangay.php logic)
        
        Args:
            days_back: Number of days to look back
            farm_id: Optional farm_parcels_id to filter by farm (like pest_class_percentage.php)
            barangay: Optional barangay name to filter by barangay (like pest_reports_barangay.php)
        """
        try:
            connection = pymysql.connect(**self.db_config)
            
            # Build query based on filters (matching report queries)
            if farm_id and farm_id > 0:
                # Farm-specific query (like pest_class_percentage.php)
                query = """
                SELECT 
                    DATE(ii.created_at) as date,
                    ii.classification_json,
                    ii.device_id,
                    d.farm_parcels_id
                FROM images_inbox ii
                INNER JOIN devices d ON d.device_id = ii.device_id
                WHERE ii.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                AND ii.classification_json IS NOT NULL 
                AND ii.classification_json != ''
                AND d.farm_parcels_id = %s
                ORDER BY ii.created_at ASC
                """
                params = [days_back, farm_id]
            elif barangay:
                # Barangay-specific query (like pest_reports_barangay.php)
                query = """
                SELECT 
                    DATE(ii.created_at) as date,
                    ii.classification_json,
                    ii.device_id,
                    pr.Barangay,
                    fp.farm_parcels_id
                FROM images_inbox ii
                INNER JOIN devices d ON d.device_id = ii.device_id
                LEFT JOIN farm_parcels fp ON fp.farm_parcels_id = d.farm_parcels_id
                LEFT JOIN profile pr ON pr.profile_id = fp.profile_id
                WHERE ii.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                AND ii.classification_json IS NOT NULL 
                AND ii.classification_json != ''
                AND pr.Barangay = %s
                ORDER BY ii.created_at ASC
                """
                params = [days_back, barangay]
            else:
                # All data query
                query = """
                SELECT 
                    DATE(created_at) as date,
                    classification_json,
                    device_id
                FROM images_inbox 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                AND classification_json IS NOT NULL 
                AND classification_json != ''
                ORDER BY created_at ASC
                """
                params = [days_back]
            
            df = pd.read_sql(query, connection, params=params)
            connection.close()
            
            # Parse pest counts from classification_json (matching report logic)
            pest_data = []
            for _, row in df.iterrows():
                try:
                    classification = json.loads(row['classification_json'])
                    
                    # Handle both formats: direct pest_counts and predictions.pest_counts
                    pest_counts = {}
                    if 'pest_counts' in classification:
                        pest_counts = classification['pest_counts']
                    elif 'predictions' in classification and 'pest_counts' in classification['predictions']:
                        pest_counts = classification['predictions']['pest_counts']
                    
                    if not pest_counts:
                        continue
                    
                    # Normalize pest type names using mapping
                    normalized_counts = {}
                    for pest_type, count in pest_counts.items():
                        # Map to standard format
                        normalized_type = self.pest_type_mapping.get(pest_type, pest_type)
                        if normalized_type in self.pest_types:
                            normalized_counts[normalized_type] = normalized_counts.get(normalized_type, 0) + count
                    
                    if not normalized_counts:
                        continue
                    
                    # Calculate total pests detected
                    total_pests = sum(normalized_counts.values())
                    
                    # Add individual pest counts
                    pest_record = {
                        'date': row['date'],
                        'total_pests': total_pests,
                        'device_id': row.get('device_id')
                    }
                    
                    # Add farm_id or barangay if available
                    if 'farm_parcels_id' in row:
                        pest_record['farm_parcels_id'] = row['farm_parcels_id']
                    if 'Barangay' in row:
                        pest_record['barangay'] = row['Barangay']
                    
                    # Add counts for each pest type
                    for pest_type in self.pest_types:
                        pest_record[f'{pest_type}_count'] = normalized_counts.get(pest_type, 0)
                    
                    pest_data.append(pest_record)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Error parsing classification_json: {e}")
                    continue
            
            if pest_data:
                return pd.DataFrame(pest_data)
            else:
                # Return empty DataFrame with correct columns
                columns = ['date', 'total_pests', 'device_id'] + [f'{pest}_count' for pest in self.pest_types]
                if farm_id:
                    columns.append('farm_parcels_id')
                if barangay:
                    columns.append('barangay')
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

    def prepare_training_data(self, farm_id: int = None, barangay: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data by combining pest and weather data
        For barangay-level: Aggregates all farms/devices in the barangay (like pest_reports_barangay.php)
        """
        logger.info(f"Preparing training data... (farm_id={farm_id}, barangay={barangay})")
        
        # Get historical data (with optional filters)
        pest_data = self.get_historical_pest_data(30, farm_id=farm_id, barangay=barangay)
        weather_data = self.get_historical_weather_data(30)
        
        if pest_data.empty or weather_data.empty:
            logger.warning("Insufficient historical data for training")
            return pd.DataFrame(), pd.DataFrame()
        
        # For barangay-level: Aggregate all data by date (combine all farms/devices in barangay)
        if barangay and 'barangay' in pest_data.columns:
            logger.info(f"Aggregating all farms/devices in barangay '{barangay}' by date...")
            # Group by date and sum all pest counts (aggregate all farms in barangay)
            date_columns = ['date'] + [f'{pest}_count' for pest in self.pest_types]
            aggregated_data = pest_data.groupby('date')[date_columns].agg({
                'date': 'first',
                **{f'{pest}_count': 'sum' for pest in self.pest_types}
            }).reset_index()
            
            # Recalculate total_pests after aggregation
            aggregated_data['total_pests'] = aggregated_data[[f'{pest}_count' for pest in self.pest_types]].sum(axis=1)
            
            pest_data = aggregated_data
        
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
            
            # Pest targets (total pests detected - aggregated for barangay if applicable)
            targets.append(row['total_pests'])
        
        X = pd.DataFrame(features, columns=[
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloudiness', 'max_temp', 'min_temp'
        ])
        
        y = pd.Series(targets, name='total_pests')
        
        logger.info(f"Prepared {len(X)} training samples")
        if barangay:
            logger.info(f"Data aggregated from all farms/devices in barangay '{barangay}'")
        return X, y

    def train_models(self, farm_id: int = None, barangay: str = None):
        """
        Train forecasting models
        Supports farm-specific or barangay-specific training
        """
        logger.info(f"Training pest forecasting models... (farm_id={farm_id}, barangay={barangay})")
        
        X, y = self.prepare_training_data(farm_id=farm_id, barangay=barangay)
        
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

    def generate_forecast(self, days_ahead: int = 7, farm_id: int = None, barangay: str = None) -> Dict:
        """
        Generate pest forecast for next N days
        For barangay-level: Aggregates all data from barangay and makes a single conclusion
        
        Args:
            days_ahead: Number of days to forecast
            farm_id: Optional farm_parcels_id for farm-specific forecast
            barangay: Optional barangay name - aggregates ALL farms/devices in barangay for conclusion
        """
        location_info = ""
        if farm_id:
            location_info = f" for Farm ID {farm_id}"
        elif barangay:
            location_info = f" for Barangay {barangay} (aggregating all farms/devices)"
        
        logger.info(f"Generating {days_ahead}-day pest forecast{location_info}...")
        
        # Get historical data for context (especially for barangay aggregation)
        historical_data = None
        if barangay:
            logger.info(f"Gathering all detection data from barangay '{barangay}'...")
            historical_data = self.get_historical_pest_data(30, barangay=barangay)
            if not historical_data.empty:
                # Aggregate by date to get barangay-level patterns
                date_columns = ['date'] + [f'{pest}_count' for pest in self.pest_types]
                aggregated = historical_data.groupby('date')[date_columns].agg({
                    'date': 'first',
                    **{f'{pest}_count': 'sum' for pest in self.pest_types}
                }).reset_index()
                aggregated['total_pests'] = aggregated[[f'{pest}_count' for pest in self.pest_types]].sum(axis=1)
                historical_data = aggregated
                logger.info(f"Found {len(historical_data)} days of aggregated data for barangay '{barangay}'")
        
        # Get weather forecast
        weather_forecast = self.get_weather_forecast(days_ahead)
        
        if weather_forecast.empty:
            logger.warning("No weather forecast available")
            return {'error': 'No weather forecast data available'}
        
        forecast_results = {
            'generated_at': datetime.now().isoformat(),
            'forecast_days': days_ahead,
            'farm_id': farm_id,
            'barangay': barangay,
            'daily_forecasts': []
        }
        
        # Add barangay summary if available
        if barangay and historical_data is not None and not historical_data.empty:
            # Calculate barangay-level statistics
            avg_daily_pests = historical_data['total_pests'].mean()
            max_daily_pests = historical_data['total_pests'].max()
            pest_totals = {pest: historical_data[f'{pest}_count'].sum() for pest in self.pest_types}
            
            forecast_results['barangay_summary'] = {
                'avg_daily_pests': round(avg_daily_pests, 2),
                'max_daily_pests': int(max_daily_pests),
                'total_days_analyzed': len(historical_data),
                'pest_totals': pest_totals
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
            
            # Generate barangay-level conclusion
            conclusion = None
            if barangay:
                conclusion = self.generate_barangay_conclusion(
                    pest_risks, weather_data, historical_data, weather_day['date']
                )
            
            day_forecast = {
                'date': weather_day['date'].strftime('%Y-%m-%d'),
                'weather': weather_data,
                'overall_risk': {
                    'level': overall_level,
                    'score': round(overall_risk, 2)
                },
                'pest_risks': pest_risks,
                'recommendations': self.generate_recommendations(pest_risks, weather_data),
                'conclusion': conclusion  # Barangay-level conclusion
            }
            
            forecast_results['daily_forecasts'].append(day_forecast)
        
        return forecast_results
    
    def generate_barangay_conclusion(self, pest_risks: Dict, weather_data: Dict, 
                                     historical_data: pd.DataFrame, forecast_date) -> str:
        """
        Generate a single conclusion for the entire barangay based on:
        - Aggregated pest risks from all farms/devices
        - Weather conditions
        - Historical patterns
        """
        conclusions = []
        
        # Analyze overall risk level
        high_risk_pests = [pest for pest, risk in pest_risks.items() 
                          if risk['risk_level'] == 'high']
        medium_risk_pests = [pest for pest, risk in pest_risks.items() 
                           if risk['risk_level'] == 'medium']
        
        if high_risk_pests:
            pest_names = ', '.join([pest.replace('_', ' ').title() for pest in high_risk_pests])
            conclusions.append(f"High pest risk detected for {pest_names} across the barangay.")
            conclusions.append("All farms in this barangay should prepare preventive measures.")
        
        if medium_risk_pests and not high_risk_pests:
            pest_names = ', '.join([pest.replace('_', ' ').title() for pest in medium_risk_pests])
            conclusions.append(f"Moderate pest activity expected for {pest_names}.")
            conclusions.append("Farmers should monitor their fields closely.")
        
        # Weather-based conclusion
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 70)
        
        if temp >= 28 and humidity >= 75:
            conclusions.append("Weather conditions are highly favorable for pest activity across the barangay.")
        elif temp >= 25 and humidity >= 70:
            conclusions.append("Weather conditions are moderately favorable for pest activity.")
        
        # Historical context
        if historical_data is not None and not historical_data.empty:
            avg_pests = historical_data['total_pests'].mean()
            if avg_pests > 10:
                conclusions.append(f"Historical data shows average of {avg_pests:.1f} pests per day in this barangay.")
                conclusions.append("Current forecast aligns with historical patterns.")
        
        if not conclusions:
            conclusions.append("Overall pest risk is low for this barangay.")
            conclusions.append("Continue regular monitoring.")
        
        return " ".join(conclusions)

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
        """
        Save forecast results to database
        Supports farm-specific and barangay-specific forecasts
        """
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Create forecast table if it doesn't exist (with location support)
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
                farm_parcels_id INT NULL,
                barangay VARCHAR(100) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_forecast_date (forecast_date),
                INDEX idx_pest_type (pest_type),
                INDEX idx_farm (farm_parcels_id),
                INDEX idx_barangay (barangay)
            )
            """
            
            cursor.execute(create_table_query)
            
            # Get location info from forecast_data
            farm_id = forecast_data.get('farm_id')
            barangay = forecast_data.get('barangay')
            
            # Insert forecast data
            for day_forecast in forecast_data.get('daily_forecasts', []):
                forecast_date = day_forecast['date']
                weather = day_forecast['weather']
                
                for pest_type, risk_info in day_forecast['pest_risks'].items():
                    insert_query = """
                    INSERT INTO pest_forecasts 
                    (forecast_date, pest_type, risk_level, risk_score, confidence,
                     weather_temperature, weather_humidity, weather_rainfall, recommendations,
                     farm_parcels_id, barangay)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        json.dumps(day_forecast['recommendations']),
                        farm_id,
                        barangay
                    ))
            
            connection.commit()
            connection.close()
            
            location_info = ""
            if farm_id:
                location_info = f" for Farm ID {farm_id}"
            elif barangay:
                location_info = f" for Barangay {barangay}"
            
            logger.info(f"Forecast data saved to database{location_info}")
            
        except Exception as e:
            logger.error(f"Error saving forecast to database: {e}")

    def get_all_barangays(self) -> List[str]:
        """Get list of all barangays with detection data (matching pest_reports_barangay.php)"""
        try:
            connection = pymysql.connect(**self.db_config)
            
            query = """
            SELECT DISTINCT pr.Barangay
            FROM devices d
            LEFT JOIN farm_parcels fp ON fp.farm_parcels_id = d.farm_parcels_id
            LEFT JOIN profile pr ON pr.profile_id = fp.profile_id
            INNER JOIN images_inbox ii ON ii.device_id = d.device_id
            WHERE ii.classification_json IS NOT NULL 
            AND ii.classification_json != ''
            AND pr.Barangay IS NOT NULL
            AND pr.Barangay != ''
            GROUP BY pr.Barangay
            HAVING COUNT(DISTINCT ii.ID) > 0
            ORDER BY pr.Barangay ASC
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            barangays = df['Barangay'].tolist() if not df.empty else []
            logger.info(f"Found {len(barangays)} barangays with detection data")
            return barangays
            
        except Exception as e:
            logger.error(f"Error getting barangays: {e}")
            return []

def main():
    """
    Main function to generate pest forecast
    Focuses on barangay-level predictions (aggregates all farms/devices in barangay)
    
    Usage:
        python pest_forecasting_engine.py                    # All barangays
        python pest_forecasting_engine.py --barangay="Brgy1" # Specific barangay
        python pest_forecasting_engine.py --farm=5           # Farm ID 5 (optional)
    """
    import sys
    
    print("🌾 AgriShield Pest Forecasting Engine")
    print("=" * 50)
    print("📍 Barangay-Level Forecasting (aggregates all farms/devices)")
    print("=" * 50)
    
    # Parse command line arguments
    farm_id = None
    barangay = None
    all_barangays = False
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--farm='):
                farm_id = int(arg.split('=')[1])
            elif arg.startswith('--barangay='):
                barangay = arg.split('=')[1]
            elif arg == '--all-barangays':
                all_barangays = True
    else:
        # Default: generate for all barangays
        all_barangays = True
    
    # Initialize forecasting engine
    engine = PestForecastingEngine()
    
    if all_barangays:
        # Get all barangays and generate forecast for each
        barangays = engine.get_all_barangays()
        
        if not barangays:
            print("❌ No barangays found with detection data")
            return
        
        print(f"\n📊 Found {len(barangays)} barangays with detection data")
        print("🔮 Generating forecasts for all barangays...\n")
        
        for brgy in barangays:
            print(f"\n{'='*60}")
            print(f"📍 Processing Barangay: {brgy}")
            print(f"{'='*60}")
            
            # Train models for this barangay
            engine.train_models(barangay=brgy)
            
            # Generate forecast
            forecast = engine.generate_forecast(7, barangay=brgy)
            
            if 'error' in forecast:
                print(f"❌ Error for {brgy}: {forecast['error']}")
                continue
            
            # Display summary
            print(f"\n✅ Forecast generated for {brgy}")
            if 'barangay_summary' in forecast:
                summary = forecast['barangay_summary']
                print(f"   📊 Avg daily pests: {summary['avg_daily_pests']}")
                print(f"   📈 Max daily pests: {summary['max_daily_pests']}")
                print(f"   📅 Days analyzed: {summary['total_days_analyzed']}")
            
            # Show first day conclusion
            if forecast['daily_forecasts']:
                first_day = forecast['daily_forecasts'][0]
                if first_day.get('conclusion'):
                    print(f"   💡 Conclusion: {first_day['conclusion'][:100]}...")
            
            # Save to database
            engine.save_forecast_to_database(forecast)
            print(f"   💾 Saved to database")
        
        print(f"\n✅ Completed forecasts for {len(barangays)} barangays")
        
    else:
        # Single location forecast
        location_info = ""
        if barangay:
            location_info = f" for Barangay {barangay} (aggregating all farms/devices)"
            print(f"🔮 Generating 7-day pest forecast{location_info}...")
        elif farm_id:
            location_info = f" for Farm ID {farm_id}"
            print(f"🔮 Generating 7-day pest forecast{location_info}...")
        else:
            print("🔮 Generating 7-day pest forecast (all locations)...")
        
        # Train models
        engine.train_models(farm_id=farm_id, barangay=barangay)
        
        # Train models
        engine.train_models(farm_id=farm_id, barangay=barangay)
        
        # Generate forecast
        forecast = engine.generate_forecast(7, farm_id=farm_id, barangay=barangay)
        
        if 'error' in forecast:
            print(f"❌ Error: {forecast['error']}")
            return
        
        # Display results
        print(f"\n📊 Pest Forecast Generated: {forecast['generated_at']}")
        print(f"📅 Forecast Period: {forecast['forecast_days']} days")
        
        if barangay and 'barangay_summary' in forecast:
            summary = forecast['barangay_summary']
            print(f"\n📍 Barangay Summary ({barangay}):")
            print(f"   📊 Avg daily pests: {summary['avg_daily_pests']}")
            print(f"   📈 Max daily pests: {summary['max_daily_pests']}")
            print(f"   📅 Days analyzed: {summary['total_days_analyzed']}")
        
        for day in forecast['daily_forecasts']:
            print(f"\n📅 {day['date']}:")
            print(f"   🌡️ Temperature: {day['weather']['temperature']:.1f}°C")
            print(f"   💧 Humidity: {day['weather']['humidity']:.1f}%")
            print(f"   🌧️ Rainfall: {day['weather']['rainfall']:.1f}mm")
            print(f"   ⚠️ Overall Risk: {day['overall_risk']['level'].upper()} ({day['overall_risk']['score']})")
            
            print(f"   🐛 Pest Risks:")
            for pest, risk in day['pest_risks'].items():
                print(f"      - {pest}: {risk['risk_level']} ({risk['risk_score']})")
            
            # Show barangay conclusion if available
            if day.get('conclusion'):
                print(f"   💡 Barangay Conclusion:")
                print(f"      {day['conclusion']}")
            
            print(f"   💡 Recommendations:")
            for rec in day['recommendations'][:3]:  # Show first 3 recommendations
                print(f"      • {rec}")
        
        # Save to database
        engine.save_forecast_to_database(forecast)
        
        print(f"\n✅ Forecast complete! Data saved to database.")
        print(f"📊 Use the API endpoints to access forecast data.")

if __name__ == "__main__":
    main()





