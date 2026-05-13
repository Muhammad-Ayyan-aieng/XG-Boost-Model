# api/models/request_response.py
from pydantic import BaseModel, Field, field_validator
from typing import Dict


class AccidentInput(BaseModel):
    # Weather
    temperature: float = Field(65.0, ge=-50, le=130)
    humidity: float = Field(60.0, ge=0, le=100)
    precipitation: float = Field(0.0, ge=0, le=10)
    visibility: float = Field(10.0, ge=0, le=12)
    wind_speed: float = Field(5.0, ge=0, le=100)
    
    # Time
    hour: int = Field(12, ge=0, le=23)
    month: int = Field(6, ge=1, le=12)
    day_of_week: int = Field(3, ge=0, le=6)
    is_weekend: int = Field(0, ge=0, le=1)
    
    # Binary features
    junction: int = Field(0, ge=0, le=1)
    traffic_signal: int = Field(0, ge=0, le=1)
    crossing: int = Field(0, ge=0, le=1)
    railway: int = Field(0, ge=0, le=1)
    stop: int = Field(0, ge=0, le=1)
    traffic_calming: int = Field(0, ge=0, le=1)
    roundabout: int = Field(0, ge=0, le=1)
    amenity: int = Field(0, ge=0, le=1)
    station: int = Field(0, ge=0, le=1)
    no_exit: int = Field(0, ge=0, le=1)
    
    # Text keywords
    has_blocked: int = Field(0, ge=0, le=1)
    has_jackknife: int = Field(0, ge=0, le=1)
    has_multi_vehicle: int = Field(0, ge=0, le=1)
    has_rollover: int = Field(0, ge=0, le=1)
    has_road_closed: int = Field(0, ge=0, le=1)
    has_slow_traffic: int = Field(0, ge=0, le=1)
    has_queueing: int = Field(0, ge=0, le=1)
    has_shoulder: int = Field(0, ge=0, le=1)
    has_injury: int = Field(0, ge=0, le=1)
    has_fatality: int = Field(0, ge=0, le=1)
    
    # Categorical
    state: str = "GA"
    weather_condition: str = "FAIR"
    time_of_day: str = "Afternoon"
    season: str = "Summer"
    sunrise_sunset: str = "Day"
    
    duration_minutes: float = Field(0, ge=0)
    
    # =========================================================
    # TYPE VALIDATORS (Prevent non-numeric values)
    # =========================================================
    
    @field_validator('temperature', mode='before')
    def check_temperature_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Temperature must be a number, got "{v}"')
        return v
    
    @field_validator('humidity', mode='before')
    def check_humidity_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Humidity must be a number, got "{v}"')
        return v
    
    @field_validator('precipitation', mode='before')
    def check_precipitation_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Precipitation must be a number, got "{v}"')
        return v
    
    @field_validator('visibility', mode='before')
    def check_visibility_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Visibility must be a number, got "{v}"')
        return v
    
    @field_validator('wind_speed', mode='before')
    def check_wind_speed_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Wind speed must be a number, got "{v}"')
        return v
    
    @field_validator('hour', mode='before')
    def check_hour_type(cls, v):
        if isinstance(v, str):
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f'Hour must be a number, got "{v}"')
        return v
    
    @field_validator('month', mode='before')
    def check_month_type(cls, v):
        if isinstance(v, str):
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f'Month must be a number, got "{v}"')
        return v
    
    @field_validator('day_of_week', mode='before')
    def check_day_of_week_type(cls, v):
        if isinstance(v, str):
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f'Day of week must be a number, got "{v}"')
        return v
    
    @field_validator('is_weekend', mode='before')
    def check_is_weekend_type(cls, v):
        if isinstance(v, str):
            if v.lower() in ['true', 'yes', '1']:
                return 1
            if v.lower() in ['false', 'no', '0']:
                return 0
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f'IsWeekend must be 0 or 1, got "{v}"')
        return v
    
    @field_validator('duration_minutes', mode='before')
    def check_duration_type(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f'Duration must be a number, got "{v}"')
        return v
    
    # Binary field type validators
    @field_validator('junction', 'traffic_signal', 'crossing', 'railway', 'stop',
                     'traffic_calming', 'roundabout', 'amenity', 'station', 'no_exit',
                     'has_blocked', 'has_jackknife', 'has_multi_vehicle', 'has_rollover',
                     'has_road_closed', 'has_slow_traffic', 'has_queueing', 'has_shoulder',
                     'has_injury', 'has_fatality', mode='before')
    def check_binary_type(cls, v, info):
        if isinstance(v, str):
            if v.lower() in ['true', 'yes', '1']:
                return 1
            if v.lower() in ['false', 'no', '0']:
                return 0
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f'{info.field_name} must be 0 or 1, got "{v}"')
        
        if v not in [0, 1]:
            raise ValueError(f'{info.field_name} must be 0 or 1, got {v}')
        return v
    
    # =========================================================
    # RANGE VALIDATORS (Prevent out-of-range values)
    # =========================================================
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < -50 or v > 130:
            raise ValueError(f'Temperature must be between -50°F and 130°F, got {v}')
        return v
    
    @field_validator('humidity')
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError(f'Humidity must be between 0% and 100%, got {v}')
        return v
    
    @field_validator('precipitation')
    def validate_precipitation(cls, v):
        if v < 0 or v > 10:
            raise ValueError(f'Precipitation must be between 0 and 10 inches, got {v}')
        return v
    
    @field_validator('visibility')
    def validate_visibility(cls, v):
        if v < 0 or v > 12:
            raise ValueError(f'Visibility must be between 0 and 12 miles, got {v}')
        return v
    
    @field_validator('wind_speed')
    def validate_wind_speed(cls, v):
        if v < 0 or v > 100:
            raise ValueError(f'Wind speed must be between 0 and 100 mph, got {v}')
        return v
    
    @field_validator('hour')
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError(f'Hour must be between 0 and 23, got {v}')
        return v
    
    @field_validator('month')
    def validate_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError(f'Month must be between 1 and 12, got {v}')
        return v
    
    @field_validator('day_of_week')
    def validate_day_of_week(cls, v):
        if v < 0 or v > 6:
            raise ValueError(f'Day of week must be between 0 and 6, got {v}')
        return v
    
    @field_validator('duration_minutes')
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError(f'Duration cannot be negative, got {v}')
        if v > 1440:
            raise ValueError(f'Duration cannot exceed 1440 minutes (24 hours), got {v}')
        return v
    
    # =========================================================
    # CATEGORICAL VALIDATORS
    # =========================================================
    
    @field_validator('state')
    def validate_state(cls, v):
        valid = ['GA', 'CT', 'CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'MI', 'NJ', 'NC', 'VA', 'WA', 'MA']
        if v not in valid:
            raise ValueError(f"State must be one of {valid}")
        return v
    
    @field_validator('weather_condition')
    def validate_weather(cls, v):
        valid = ['FAIR', 'CLEAR', 'MOSTLY CLOUDY', 'CLOUDY', 'PARTLY CLOUDY', 
                 'LIGHT RAIN', 'RAIN', 'HEAVY RAIN', 'FOG', 'SNOW', 'UNKNOWN']
        if v.upper() not in valid:
            return 'UNKNOWN'
        return v.upper()
    
    @field_validator('time_of_day')
    def validate_time(cls, v):
        valid = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']
        return v if v in valid else 'Afternoon'
    
    @field_validator('season')
    def validate_season(cls, v):
        valid = ['Spring', 'Summer', 'Fall', 'Winter']
        return v if v in valid else 'Summer'
    
    @field_validator('sunrise_sunset')
    def validate_sunrise(cls, v):
        valid = ['Day', 'Night']
        return v if v in valid else 'Day'


class AccidentResponse(BaseModel):
    severity: int
    expected_severity: float
    severity_label: str
    probability: float
    confidence: str
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str