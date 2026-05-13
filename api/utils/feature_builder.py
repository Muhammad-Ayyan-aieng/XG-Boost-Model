# api/utils/feature_builder.py
"""
Converts user input to 43-feature array for model prediction
"""

from config import (
    FEATURE_NAMES, DEFAULT_VALUES,
    STATE_MAP, WEATHER_MAP, TIME_MAP, SEASON_MAP, SUNRISE_MAP
)


def build_features(input_data) -> list:
    """
    Convert AccidentInput to 43-number list for model prediction.
    
    Args:
        input_data: AccidentInput object from request
        
    Returns:
        List of 43 floats/ints in the exact order the model expects
    """
    # Start with default values
    features = DEFAULT_VALUES.copy()
    
    # =========================================================
    # Override with user provided values
    # =========================================================
    
    # Weather features
    features['Temperature(F)'] = input_data.temperature
    features['Humidity(%)'] = input_data.humidity
    features['Precipitation(in)'] = input_data.precipitation
    features['Visibility(mi)'] = input_data.visibility
    features['Wind_Speed(mph)'] = input_data.wind_speed
    
    # Time features
    features['Hour'] = input_data.hour
    features['Month'] = input_data.month
    features['DayOfWeek'] = input_data.day_of_week
    features['IsWeekend'] = input_data.is_weekend
    
    # Road features (binary)
    features['Junction'] = input_data.junction
    features['Traffic_Signal'] = input_data.traffic_signal
    features['Crossing'] = input_data.crossing
    features['Railway'] = input_data.railway
    features['Stop'] = input_data.stop
    features['Traffic_Calming'] = input_data.traffic_calming
    features['Roundabout'] = input_data.roundabout
    features['Amenity'] = input_data.amenity
    features['Station'] = input_data.station
    features['No_Exit'] = input_data.no_exit
    
    # Text keyword features (binary)
    features['has_blocked'] = input_data.has_blocked
    features['has_jackknife'] = input_data.has_jackknife
    features['has_multi_vehicle'] = input_data.has_multi_vehicle
    features['has_rollover'] = input_data.has_rollover
    features['has_road_closed'] = input_data.has_road_closed
    features['has_slow_traffic'] = input_data.has_slow_traffic
    features['has_queueing'] = input_data.has_queueing
    features['has_shoulder'] = input_data.has_shoulder
    features['has_injury'] = input_data.has_injury
    features['has_fatality'] = input_data.has_fatality
    
    # Categorical features (convert text to numbers)
    features['State'] = STATE_MAP.get(input_data.state, 9)
    features['Weather_Condition'] = WEATHER_MAP.get(input_data.weather_condition, 1)
    features['TimeOfDay'] = TIME_MAP.get(input_data.time_of_day, 2)
    features['Season'] = SEASON_MAP.get(input_data.season, 1)
    features['Sunrise_Sunset'] = SUNRISE_MAP.get(input_data.sunrise_sunset, 0)
    
    # Optional features
    features['Duration_Minutes'] = input_data.duration_minutes
    
    # Return in the exact order the model expects
    return [features[name] for name in FEATURE_NAMES]