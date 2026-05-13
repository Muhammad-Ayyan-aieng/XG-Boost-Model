# api/config.py
"""
Configuration - Full feature list with default values
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs/models/final_model.pkl")

API_TITLE = "Accident Severity Predictor API"
API_VERSION = "1.0.0"
HOST = "0.0.0.0"
PORT = 8000

# =========================================================
# COMPLETE FEATURE LIST (43 features in correct order)
# =========================================================

FEATURE_NAMES = [
    'Temperature(F)', 'Humidity(%)', 'Precipitation(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Junction', 'Traffic_Signal', 'Crossing', 'Railway', 'Stop',
    'Traffic_Calming', 'Roundabout', 'Amenity', 'Station', 'No_Exit',
    'has_jackknife', 'has_blocked', 'has_multi_vehicle', 'has_rollover',
    'has_road_closed', 'has_slow_traffic', 'has_queueing', 'has_shoulder',
    'has_fatality', 'has_injury', 'has_ice', 'has_snow', 'has_fog', 'has_heavy_rain',
    'State', 'Weather_Condition', 'TimeOfDay', 'Season', 'Sunrise_Sunset',
    'Rain_Junction', 'Rush_Hour', 'Late_Night'
]

# =========================================================
# DEFAULT VALUES (used when user doesn't provide)
# =========================================================

DEFAULT_VALUES = {
    'Temperature(F)': 61.42,      # mean from data
    'Humidity(%)': 68.37,         # mean from data
    'Precipitation(in)': 0.01,    # mean from data
    'Visibility(mi)': 8.98,       # mean from data
    'Wind_Speed(mph)': 7.01,      # mean from data
    'Hour': 12,
    'Month': 6,
    'DayOfWeek': 3,
    'IsWeekend': 0,
    'Junction': 0,
    'Traffic_Signal': 0,
    'Crossing': 0,
    'Railway': 0,
    'Stop': 0,
    'Traffic_Calming': 0,
    'Roundabout': 0,
    'Amenity': 0,
    'Station': 0,
    'No_Exit': 0,
    'has_jackknife': 0,
    'has_blocked': 0,
    'has_multi_vehicle': 0,
    'has_rollover': 0,
    'has_road_closed': 0,
    'has_slow_traffic': 0,
    'has_queueing': 0,
    'has_shoulder': 0,
    'has_fatality': 0,
    'has_injury': 0,
    'has_ice': 0,
    'has_snow': 0,
    'has_fog': 0,
    'has_heavy_rain': 0,
    'State': 9,                   # GA = 9
    'Weather_Condition': 1,       # FAIR = 1
    'TimeOfDay': 2,               # Afternoon = 2
    'Season': 1,                  # Summer = 1
    'Sunrise_Sunset': 0,          # Day = 0
    'Rain_Junction': 0,
    'Rush_Hour': 0,
    'Late_Night': 0,
}

# =========================================================
# CATEGORICAL MAPS (for user input → numbers)
# =========================================================

STATE_MAP = {
    'GA': 9, 'CT': 6, 'CA': 4, 'TX': 42, 'FL': 8, 'NY': 31,
    'IL': 12, 'PA': 37, 'OH': 34, 'MI': 21, 'NJ': 29, 'NC': 32,
    'VA': 45, 'WA': 46, 'MA': 20, 'AZ': 2, 'TN': 41, 'IN': 13,
    'MO': 24, 'MD': 19, 'WI': 48, 'CO': 5, 'MN': 22, 'SC': 39,
    'AL': 0, 'LA': 17, 'KY': 16, 'OR': 36, 'OK': 35
}

WEATHER_MAP = {
    'FAIR': 1, 'CLEAR': 2, 'MOSTLY CLOUDY': 3, 'CLOUDY': 4,
    'PARTLY CLOUDY': 5, 'LIGHT RAIN': 6, 'RAIN': 7, 'HEAVY RAIN': 8,
    'FOG': 9, 'SNOW': 12, 'UNKNOWN': 0
}

TIME_MAP = {
    'Late Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4
}

SEASON_MAP = {
    'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3
}

SUNRISE_MAP = {
    'Day': 0, 'Night': 1
}