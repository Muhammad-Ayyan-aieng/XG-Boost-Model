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
    severity_label: str
    probability: float
    confidence: str
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str