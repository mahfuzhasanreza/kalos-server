from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

class CoordinatesRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    elevation: Optional[float] = Field(0, description="Elevation in meters")

class LocationRequest(BaseModel):
    location: str = Field(..., description="Location name (e.g., 'Paris, France')")
    elevation: Optional[float] = Field(0, description="Elevation in meters")
    source: Optional[str] = Field("auto", description="Data source: nasa, ai, or auto")
    location_type: Optional[str] = Field("general", description="Location type: beach, mountain, city, forest, general")
    forecast_days: Optional[int] = Field(7, ge=1, le=30, description="Number of forecast days")

class WeatherData(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    precipitation_probability: float = Field(..., description="Precipitation probability percentage")
    weather_condition: str = Field(..., description="Weather condition")
    pressure: Optional[float] = Field(1013.25, description="Atmospheric pressure in hPa")
    source: str = Field(..., description="Data source")
    timestamp: str = Field(..., description="Data timestamp")
    confidence: int = Field(..., description="Prediction confidence percentage")
    model_used: Optional[bool] = Field(False, description="Whether AI model was used")
    nasa_data_used: Optional[bool] = Field(False, description="Whether NASA data was used")
    precipitation: Optional[float] = Field(None, description="Recent precipitation in mm")

class TravelRecommendation(BaseModel):
    rating: str = Field(..., description="Travel recommendation rating")
    score: int = Field(..., description="Comfort score (0-100)")
    recommendation: str = Field(..., description="Travel recommendation text")
    details: Dict[str, str] = Field(..., description="Detailed recommendation breakdown")

class WeatherResponse(BaseModel):
    location: str = Field(..., description="Location name")
    coordinates: Dict[str, float] = Field(..., description="Coordinates and elevation")
    weather: WeatherData = Field(..., description="Weather data")
    probabilities: Dict[str, float] = Field(..., description="Weather condition probabilities")
    travel_recommendation: TravelRecommendation = Field(..., description="Travel recommendation")
    timestamp: str = Field(..., description="Response timestamp")

class ForecastData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    precipitation_probability: float
    weather_condition: str
    pressure: Optional[float] = 1013.25
    source: str
    timestamp: str
    confidence: int
    model_used: Optional[bool] = False
    nasa_data_used: Optional[bool] = False
    date: str
    day_name: str

class ForecastResponse(BaseModel):
    location: str = Field(..., description="Location name")
    coordinates: Dict[str, float] = Field(..., description="Coordinates and elevation")
    forecast_days: int = Field(..., description="Number of forecast days")
    forecasts: List[ForecastData] = Field(..., description="List of daily forecasts")
    timestamp: str = Field(..., description="Response timestamp")

class SystemMetricsResponse(BaseModel):
    model_metrics: Dict[str, Any] = Field(..., description="AI model performance metrics")
    system_info: Dict[str, Any] = Field(..., description="System information")
    is_trained: bool = Field(..., description="Whether models are trained")
    timestamp: str = Field(..., description="Response timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")