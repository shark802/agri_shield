#!/usr/bin/env python3
"""
AgriShield Pest Forecasting Engine (Deployment)
Uses historical pest detection data (images_inbox -> devices -> farm_parcels)
Outputs percentage-based risk per pest; no weather dependency.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pymysql

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PestForecastingEngine:
    def __init__(self, db_config: Dict = None):
        """
        Initialize forecasting engine (deployment version)
        - Uses env vars when available for Heroku-style deployment.
        """
        if db_config:
            self.db_config = db_config
        else:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'database': os.getenv('DB_NAME', 'asdb'),
                'charset': os.getenv('DB_CHARSET', 'utf8mb4')
            }

        # Standard pest types (aligned with detection)
        self.pest_types = [
            'Rice_Bug',
            'green_hopper',
            'black-bug',
            'brown_hopper',
            'White_stem_borer'
        ]

        # Map alternative names to standard names
        self.pest_type_mapping = {
            'rice_bug': 'Rice_Bug',
            'ricebug': 'Rice_Bug',
            'green_leaf_hopper': 'green_hopper',
            'green_leaff_hopper': 'green_hopper',
            'greenleafhopper': 'green_hopper',
            'black_bug': 'black-bug',
            'blackbug': 'black-bug',
            'brown_plant_hopper': 'brown_hopper',
            'brownplanthopper': 'brown_hopper',
            'white_stem_borer': 'White_stem_borer',
            'white_stemborer': 'White_stem_borer'
        }

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    def get_historical_pest_data(self, days_back: int = 30, farm_id: int = None, barangay: str = None) -> pd.DataFrame:
        """
        Fetch historical pest detection data from images_inbox, joined through devices -> farm_parcels (barangay).
        """
        try:
            conn = pymysql.connect(**self.db_config)
            if farm_id and farm_id > 0:
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
                # Barangay is stored as farm_location in farm_parcels; fallback to profile.Barangay
                query = """
                SELECT 
                    DATE(ii.created_at) as date,
                    ii.classification_json,
                    ii.device_id,
                    COALESCE(fp.farm_location, pr.Barangay) as Barangay,
                    fp.farm_parcels_id
                FROM images_inbox ii
                INNER JOIN devices d ON d.device_id = ii.device_id
                LEFT JOIN farm_parcels fp ON fp.farm_parcels_id = d.farm_parcels_id
                LEFT JOIN profile pr ON pr.profile_id = fp.profile_id
                WHERE ii.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                  AND ii.classification_json IS NOT NULL 
                  AND ii.classification_json != ''
                  AND (fp.farm_location = %s OR (fp.farm_location IS NULL AND pr.Barangay = %s))
                ORDER BY ii.created_at ASC
                """
                params = [days_back, barangay, barangay]
            else:
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

            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting historical pest data: {e}")
            return pd.DataFrame()

    def parse_pest_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse pest_counts from classification_json and normalize pest names.
        """
        if df.empty:
            return pd.DataFrame()

        parsed_rows = []
        for _, row in df.iterrows():
            try:
                cls = json.loads(row['classification_json'])
            except Exception:
                continue

            pest_counts = {}
            if isinstance(cls, dict):
                if 'pest_counts' in cls:
                    pest_counts = cls.get('pest_counts', {})
                elif 'predictions' in cls and isinstance(cls['predictions'], dict):
                    pest_counts = cls['predictions'].get('pest_counts', {})

            if not pest_counts:
                continue

            normalized = {}
            for p, c in pest_counts.items():
                mapped = self.pest_type_mapping.get(p, p)
                if mapped in self.pest_types:
                    normalized[mapped] = normalized.get(mapped, 0) + c

            if not normalized:
                continue

            total = sum(normalized.values())
            pest_record = {
                'date': row.get('date'),
                'device_id': row.get('device_id'),
                'farm_parcels_id': row.get('farm_parcels_id'),
                'Barangay': row.get('Barangay'),
                'total_pests': total
            }
            for p in self.pest_types:
                pest_record[p] = normalized.get(p, 0)

            parsed_rows.append(pest_record)

        return pd.DataFrame(parsed_rows)

    def predict_pest_risk_from_history(self, days_back: int = 30, farm_id: int = None, barangay: str = None) -> Dict:
        """
        Compute percentage-based pest risk using historical detection data only.
        """
        raw_df = self.get_historical_pest_data(days_back=days_back, farm_id=farm_id, barangay=barangay)
        parsed = self.parse_pest_counts(raw_df)
        if parsed.empty:
            return {}

        totals_by_pest = {p: 0 for p in self.pest_types}
        for _, row in parsed.iterrows():
            for p in self.pest_types:
                totals_by_pest[p] += row.get(p, 0)

        overall_total = sum(totals_by_pest.values())
        if overall_total <= 0:
            return {}

        risks = {}
        for pest, count in totals_by_pest.items():
            pct = (count / overall_total) * 100.0
            if pct >= 40:
                level = 'high'
            elif pct >= 20:
                level = 'medium'
            else:
                level = 'low'
            risks[pest] = {
                'percentage': round(pct, 2),
                'risk_level': level,
                'risk_score_decimal': round(pct / 100.0, 3),
                'confidence': 0.80
            }
        return risks

    def generate_forecast(self, days_ahead: int = 7, farm_id: int = None, barangay: str = None) -> Dict:
        """
        Generate a simple forecast by projecting recent pest percentages across the horizon.
        """
        pest_risks = self.predict_pest_risk_from_history(days_back=30, farm_id=farm_id, barangay=barangay)
        if not pest_risks:
            return {'error': 'No pest detection data available'}

        today = datetime.utcnow().date()
        daily_forecasts = []
        for i in range(days_ahead):
            date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            daily_forecasts.append({
                'date': date_str,
                'pest_risks': pest_risks,
                'recommendations': self.generate_recommendations(pest_risks)
            })

        return {
            'generated_at': datetime.utcnow().isoformat(),
            'forecast_days': days_ahead,
            'daily_forecasts': daily_forecasts,
            'barangay': barangay,
            'farm_id': farm_id
        }

    def generate_recommendations(self, pest_risks: Dict) -> List[str]:
        """
        Simple recommendations based on highest risk pests.
        """
        recs = []
        high = [p for p, r in pest_risks.items() if r['risk_level'] == 'high']
        medium = [p for p, r in pest_risks.items() if r['risk_level'] == 'medium']

        if high:
            recs.append(f"High outbreak risk detected: {', '.join(high)}. Prepare immediate control measures.")
        if medium:
            recs.append(f"Monitor and apply preventive measures for: {', '.join(medium)}.")
        if not recs:
            recs.append("Low outbreak risk. Continue regular monitoring and good field hygiene.")
        return recs

    def save_forecast_to_database(self, forecast_data: Dict):
        """
        Save forecast results to pest_forecasts table.
        Stores percentage as decimal in risk_score.
        """
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            create_table = """
            CREATE TABLE IF NOT EXISTS pest_forecasts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                forecast_date DATE NOT NULL,
                pest_type VARCHAR(50) NOT NULL,
                risk_level ENUM('low', 'medium', 'high') NOT NULL,
                risk_score DECIMAL(3,2) NOT NULL,
                confidence DECIMAL(3,2) NOT NULL DEFAULT 0.80,
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
            cursor.execute(create_table)
            
            # Check and add missing columns if table already exists
            try:
                cursor.execute("SHOW COLUMNS FROM pest_forecasts LIKE 'farm_parcels_id'")
                if cursor.fetchone() is None:
                    cursor.execute("ALTER TABLE pest_forecasts ADD COLUMN farm_parcels_id INT NULL")
                    cursor.execute("ALTER TABLE pest_forecasts ADD INDEX idx_farm (farm_parcels_id)")
            except Exception:
                pass  # Column might already exist or index might exist
            
            try:
                cursor.execute("SHOW COLUMNS FROM pest_forecasts LIKE 'barangay'")
                if cursor.fetchone() is None:
                    cursor.execute("ALTER TABLE pest_forecasts ADD COLUMN barangay VARCHAR(100) NULL")
                    cursor.execute("ALTER TABLE pest_forecasts ADD INDEX idx_barangay (barangay)")
            except Exception:
                pass  # Column might already exist or index might exist

            farm_id = forecast_data.get('farm_id')
            barangay = forecast_data.get('barangay')

            for day in forecast_data.get('daily_forecasts', []):
                date_str = day['date']
                recs_json = json.dumps(day.get('recommendations', []))
                for pest, info in day.get('pest_risks', {}).items():
                    insert = """
                    INSERT INTO pest_forecasts
                    (forecast_date, pest_type, risk_level, risk_score, confidence,
                     weather_temperature, weather_humidity, weather_rainfall, recommendations,
                     farm_parcels_id, barangay)
                    VALUES (%s,%s,%s,%s,%s,NULL,NULL,NULL,%s,%s,%s)
                    """
                    cursor.execute(insert, (
                        date_str,
                        pest,
                        info['risk_level'],
                        info['risk_score_decimal'],
                        info.get('confidence', 0.80),
                        recs_json,
                        farm_id,
                        barangay
                    ))

            conn.commit()
            conn.close()
            logger.info("Forecast data saved to database")
        except Exception as e:
            logger.error(f"Error saving forecast to database: {e}")

    def get_all_barangays(self) -> List[str]:
        """
        Return barangays that have pest detection data.
        In this schema, barangay is stored as farm_parcels.farm_location.
        If farm_location is missing, fall back to profile.Barangay.
        """
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()

            # Check if farm_parcels has farm_location column
            has_fp_farm_location = False
            try:
                cursor.execute("SHOW COLUMNS FROM farm_parcels LIKE 'farm_location'")
                has_fp_farm_location = cursor.fetchone() is not None
            except Exception:
                has_fp_farm_location = False

            if has_fp_farm_location:
                query = """
                SELECT DISTINCT COALESCE(fp.farm_location, pr.Barangay) as Barangay
                FROM devices d
                LEFT JOIN farm_parcels fp ON fp.farm_parcels_id = d.farm_parcels_id
                LEFT JOIN profile pr ON pr.profile_id = fp.profile_id
                INNER JOIN images_inbox ii ON ii.device_id = d.device_id
                WHERE ii.classification_json IS NOT NULL 
                  AND ii.classification_json != ''
                  AND (fp.farm_location IS NOT NULL AND fp.farm_location != '' OR pr.Barangay IS NOT NULL AND pr.Barangay != '')
                GROUP BY COALESCE(fp.farm_location, pr.Barangay)
                HAVING COUNT(DISTINCT ii.ID) > 0
                ORDER BY COALESCE(fp.farm_location, pr.Barangay) ASC
                """
            else:
                # Fallback to profile.Barangay only
                query = """
                SELECT DISTINCT pr.Barangay as Barangay
                FROM devices d
                LEFT JOIN farm_parcels fp ON fp.farm_parcels_id = d.farm_parcels_id
                LEFT JOIN profile pr ON pr.profile_id = fp.profile_id
                INNER JOIN images_inbox ii ON ii.device_id = d.device_id
                WHERE ii.classification_json IS NOT NULL 
                  AND ii.classification_json != ''
                  AND pr.Barangay IS NOT NULL AND pr.Barangay != ''
                GROUP BY pr.Barangay
                HAVING COUNT(DISTINCT ii.ID) > 0
                ORDER BY pr.Barangay ASC
                """

            df = pd.read_sql(query, conn)
            conn.close()
            return df['Barangay'].tolist() if not df.empty else []
        except Exception as e:
            logger.error(f"Error getting barangays: {e}")
            return []


def main():
    """
    Simple CLI to generate forecasts for all barangays or a single barangay.
    """
    print("🌾 AgriShield Pest Forecasting Engine (Deployment)")
    engine = PestForecastingEngine()

    barangays = engine.get_all_barangays()
    if not barangays:
        print("No barangays with detection data found.")
        return

    print(f"Generating forecasts for {len(barangays)} barangays...")
    for brgy in barangays:
        try:
            forecast = engine.generate_forecast(days_ahead=7, barangay=brgy)
            if 'error' in forecast:
                print(f"- {brgy}: {forecast['error']}")
                continue
            engine.save_forecast_to_database(forecast)
            print(f"- {brgy}: saved {len(forecast['daily_forecasts'])} days")
        except Exception as e:
            print(f"- {brgy}: error {e}")


if __name__ == "__main__":
    main()





