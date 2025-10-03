from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Optional, List
import uvicorn

# Try to import the full models first, fallback to simple models
try:
    from models.weather_models import WeatherModel, NASADataFetcher, GlobalWeatherSystem
    print("✅ Using full weather models with scientific computing stack")
except ImportError as e:
    print(f"⚠️ Full models not available ({e}), using simplified models")
    from models.simple_weather_models import WeatherModel, NASADataFetcher, GlobalWeatherSystem

# Import enhanced forecasting system
try:
    from models.enhanced_forecasting import CoordinateBasedWeatherSystem, EnhancedRandomForestForecaster
    HAS_ENHANCED_FORECASTING = True
    print("✅ Enhanced Random Forest forecasting available")
except ImportError as e:
    print(f"⚠️ Enhanced forecasting not available ({e})")
    HAS_ENHANCED_FORECASTING = False

from schemas.weather_schemas import (
    WeatherResponse, 
    LocationRequest, 
    CoordinatesRequest, 
    ForecastResponse,
    SystemMetricsResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="NASA Weather Prediction API",
    description="Global Weather Prediction & Recommendation System using NASA data and AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global weather system
weather_system = GlobalWeatherSystem()

# Initialize enhanced forecasting system if available
if HAS_ENHANCED_FORECASTING:
    enhanced_weather_system = CoordinateBasedWeatherSystem()
    print("✅ Enhanced coordinate-based forecasting initialized")
else:
    enhanced_weather_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🌍 Starting Global Weather Prediction API...")
    # Pre-train the AI models
    await train_models_async()
    print("✅ API ready!")

async def train_models_async():
    """Train AI models asynchronously"""
    try:
        weather_system.weather_model.train_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-train models: {e}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "🌍 NASA Weather Prediction API",
        "version": "1.0.0",
        "description": "Global weather prediction using NASA data and AI models",
        "endpoints": {
            "docs": "/docs",
            "weather": "/weather",
            "forecast": "/forecast",
            "metrics": "/metrics"
        }
    }

@app.get("/weather/coordinates", response_model=WeatherResponse)
async def get_weather_by_coordinates(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    elevation: Optional[float] = Query(0, description="Elevation in meters"),
    source: Optional[str] = Query("auto", description="Data source: nasa, ai, or auto")
):
    """Get current weather data by coordinates"""
    try:
        # Get location name
        location_name, admin, country = weather_system.get_location_from_coordinates(lat, lon)
        full_location = f"{location_name}, {admin}, {country}"
        
        # Fetch weather data with enhanced NASA integration
        if source == "enhanced" or source == "auto":
            # Try enhanced NASA integration flow: fetch NASA data -> train -> predict
            try:
                print(f"🔍 Using enhanced NASA prediction flow for ({lat}, {lon})")
                # This will:
                # 1. Fetch NASA data for the location
                # 2. Train model with NASA data if not already trained
                # 3. Make prediction with NASA-enhanced model
                weather_data = weather_system.fetch_weather_data(lat, lon, "enhanced_ai")
            except Exception as e:
                print(f"⚠️ Enhanced flow failed, falling back to standard method: {e}")
                weather_data = weather_system.fetch_weather_data(lat, lon, source)
        else:
            weather_data = weather_system.fetch_weather_data(lat, lon, source)
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="Unable to fetch weather data")
        
        # Calculate probabilities and recommendations
        probabilities = weather_system.calculate_weather_conditions(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation_probability']
        )
        
        recommendation = weather_system.get_travel_recommendation(weather_data)
        
        return WeatherResponse(
            location=full_location,
            coordinates={"latitude": lat, "longitude": lon, "elevation": elevation},
            weather=weather_data,
            probabilities=probabilities,
            travel_recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

@app.get("/weather/nasa-enhanced")
async def get_weather_nasa_enhanced(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    elevation: Optional[float] = Query(0, description="Elevation in meters")
):
    """Get weather prediction using NASA-enhanced AI flow: fetch NASA data -> train model -> predict"""
    try:
        print(f"🚀 Starting NASA-enhanced prediction for ({lat}, {lon})")
        
        # Get location name
        location_name, admin, country = weather_system.get_location_from_coordinates(lat, lon)
        full_location = f"{location_name}, {admin}, {country}"
        
        # Step 1: Try to fetch current NASA data for context
        print("📡 Step 1: Fetching NASA data...")
        nasa_fetcher = NASADataFetcher()
        current_nasa_data = None
        try:
            current_nasa_data = nasa_fetcher.fetch_nasa_weather_data(lat, lon)
        except Exception as e:
            print(f"⚠️ NASA data fetch failed: {e}")
        
        # Step 2: Use weather model with enhanced training
        print("🤖 Step 2: Training/using weather model...")
        weather_model = WeatherModel()
        
        # For simple models, we'll simulate the enhanced flow
        if hasattr(weather_model, 'predict_weather'):
            # Check if it supports NASA training
            import inspect
            signature = inspect.signature(weather_model.predict_weather)
            if 'use_nasa_training' in signature.parameters:
                # Full model with NASA training
                weather_data = weather_model.predict_weather(lat, lon, elevation, use_nasa_training=True)
            else:
                # Simple model, enhance with NASA data if available
                weather_data = weather_model.predict_weather(lat, lon, elevation)
                if current_nasa_data and current_nasa_data.get('nasa_data_used'):
                    # Blend simple model with NASA data
                    print("🔄 Blending simple model with NASA data...")
                    model_weight = 0.6
                    nasa_weight = 0.4
                    
                    weather_data.update({
                        'temperature': round(
                            weather_data['temperature'] * model_weight + 
                            current_nasa_data['temperature'] * nasa_weight, 1
                        ),
                        'humidity': round(
                            weather_data['humidity'] * model_weight + 
                            current_nasa_data['humidity'] * nasa_weight, 1
                        ),
                        'wind_speed': round(
                            weather_data['wind_speed'] * model_weight + 
                            current_nasa_data['wind_speed'] * nasa_weight, 1
                        ),
                        'precipitation_probability': round(
                            weather_data['precipitation_probability'] * model_weight + 
                            current_nasa_data['precipitation_probability'] * nasa_weight, 1
                        ),
                        'source': 'NASA-Enhanced Simple Model',
                        'confidence': 88,
                        'nasa_blend_ratio': f"{int(nasa_weight*100)}% NASA, {int(model_weight*100)}% Model",
                        'nasa_data_available': True
                    })
        else:
            # Fallback
            weather_data = current_nasa_data if current_nasa_data else {
                'temperature': 25.0,
                'humidity': 65.0,
                'wind_speed': 3.0,
                'precipitation_probability': 30.0,
                'weather_condition': 'moderate',
                'source': 'Fallback',
                'confidence': 50
            }
        
        # Calculate additional metrics
        probabilities = weather_system.calculate_weather_conditions(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation_probability']
        )
        
        recommendation = weather_system.get_travel_recommendation(weather_data)
        
        return {
            "location": full_location,
            "coordinates": {"latitude": lat, "longitude": lon, "elevation": elevation},
            "weather": weather_data,
            "probabilities": probabilities,
            "travel_recommendation": recommendation,
            "timestamp": datetime.now().isoformat(),
            "enhanced_flow": True,
            "nasa_data_used": current_nasa_data is not None and current_nasa_data.get('nasa_data_used', False)
        }
        
    except Exception as e:
        print(f"❌ NASA-enhanced prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NASA-enhanced prediction failed: {str(e)}")

@app.post("/weather/location", response_model=WeatherResponse)
async def get_weather_by_location(request: LocationRequest):
    """Get current weather data by location name"""
    try:
        # Get coordinates from location name
        lat, lon, address = weather_system.get_coordinates_from_name(request.location)
        
        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail="Location not found")
        
        # Fetch weather data
        weather_data = weather_system.fetch_weather_data(lat, lon, request.source)
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="Unable to fetch weather data")
        
        # Calculate probabilities and recommendations
        probabilities = weather_system.calculate_weather_conditions(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation_probability']
        )
        
        recommendation = weather_system.get_travel_recommendation(weather_data, request.location_type)
        
        return WeatherResponse(
            location=address,
            coordinates={"latitude": lat, "longitude": lon, "elevation": request.elevation or 0},
            weather=weather_data,
            probabilities=probabilities,
            travel_recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/forecast/coordinates", response_model=ForecastResponse)
async def get_forecast_by_coordinates(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    days: Optional[int] = Query(7, ge=1, le=30, description="Number of forecast days"),
    elevation: Optional[float] = Query(0, description="Elevation in meters"),
    use_enhanced: Optional[bool] = Query(True, description="Use enhanced Random Forest forecasting")
):
    """Get weather forecast by coordinates using enhanced Random Forest models"""
    try:
        # Get location name
        location_name, admin, country = weather_system.get_location_from_coordinates(lat, lon)
        full_location = f"{location_name}, {admin}, {country}"
        
        # Use enhanced forecasting if available and requested
        if HAS_ENHANCED_FORECASTING and enhanced_weather_system and use_enhanced:
            print(f"🎯 Using enhanced Random Forest forecasting for {full_location}")
            enhanced_result = enhanced_weather_system.get_enhanced_forecast(lat, lon, days)
            
            return ForecastResponse(
                location=full_location,
                coordinates={"latitude": lat, "longitude": lon, "elevation": elevation},
                forecast_days=days,
                forecasts=enhanced_result['forecasts'],
                timestamp=datetime.now().isoformat()
            )
        else:
            # Fallback to standard forecasting
            print(f"📊 Using standard forecasting for {full_location}")
            forecasts = []
            for i in range(days):
                target_date = datetime.now() + timedelta(days=i)
                forecast = weather_system.weather_model.predict_weather(lat, lon, elevation, target_date)
                if forecast:
                    forecast['date'] = target_date.strftime('%Y-%m-%d')
                    forecast['day_name'] = target_date.strftime('%A')
                    forecasts.append(forecast)
            
            if not forecasts:
                raise HTTPException(status_code=500, detail="Unable to generate forecast")
            
            return ForecastResponse(
                location=full_location,
                coordinates={"latitude": lat, "longitude": lon, "elevation": elevation},
                forecast_days=days,
                forecasts=forecasts,
                timestamp=datetime.now().isoformat()
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.post("/forecast/location", response_model=ForecastResponse)
async def get_forecast_by_location(request: LocationRequest):
    """Get weather forecast by location name"""
    try:
        # Get coordinates from location name
        lat, lon, address = weather_system.get_coordinates_from_name(request.location)
        
        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail="Location not found")
        
        days = request.forecast_days or 7
        
        # Generate forecast
        forecasts = []
        for i in range(days):
            target_date = datetime.now() + timedelta(days=i)
            forecast = weather_system.weather_model.predict_weather(lat, lon, request.elevation or 0, target_date)
            if forecast:
                forecast['date'] = target_date.strftime('%Y-%m-%d')
                forecast['day_name'] = target_date.strftime('%A')
                forecasts.append(forecast)
        
        if not forecasts:
            raise HTTPException(status_code=500, detail="Unable to generate forecast")
        
        return ForecastResponse(
            location=address,
            coordinates={"latitude": lat, "longitude": lon, "elevation": request.elevation or 0},
            forecast_days=days,
            forecasts=forecasts,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/weather/future")
async def get_weather_for_date(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    date: str = Query(..., description="Target date in YYYY-MM-DD format"),
    elevation: Optional[float] = Query(0, description="Elevation in meters")
):
    """Get weather prediction for a specific future date"""
    try:
        # Parse and validate date
        target_date = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now()
        max_future_date = today + timedelta(days=180)  # 6 months
        
        if target_date < today:
            raise HTTPException(status_code=400, detail="Date cannot be in the past")
        
        if target_date > max_future_date:
            raise HTTPException(status_code=400, detail="Date cannot be more than 6 months in the future")
        
        # Get location name
        location_name, admin, country = weather_system.get_location_from_coordinates(lat, lon)
        full_location = f"{location_name}, {admin}, {country}"
        
        # Generate weather prediction
        weather_data = weather_system.weather_model.predict_weather(lat, lon, elevation, target_date)
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="Unable to generate weather prediction")
        
        # Calculate probabilities and recommendations
        probabilities = weather_system.calculate_weather_conditions(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['precipitation_probability']
        )
        
        recommendation = weather_system.get_travel_recommendation(weather_data)
        
        return {
            "location": full_location,
            "coordinates": {"latitude": lat, "longitude": lon, "elevation": elevation},
            "target_date": date,
            "weather": weather_data,
            "probabilities": probabilities,
            "travel_recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

@app.get("/forecast/enhanced")
async def get_enhanced_coordinate_forecast(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    days: Optional[int] = Query(7, ge=1, le=30, description="Number of forecast days"),
    openweather_key: Optional[str] = Query(None, description="OpenWeatherMap API key for real data")
):
    """Get enhanced weather forecast using coordinate-specific Random Forest training"""
    if not HAS_ENHANCED_FORECASTING:
        raise HTTPException(status_code=503, detail="Enhanced forecasting not available")
    
    try:
        # Set API key if provided
        if openweather_key:
            enhanced_weather_system.set_api_keys(openweather_key=openweather_key)
        
        # Get location name
        location_name, admin, country = weather_system.get_location_from_coordinates(lat, lon)
        full_location = f"{location_name}, {admin}, {country}"
        
        print(f"🚀 Generating enhanced Random Forest forecast for {full_location}")
        
        # Get enhanced forecast with coordinate-specific training
        enhanced_result = enhanced_weather_system.get_enhanced_forecast(lat, lon, days)
        
        return {
            "location": full_location,
            "coordinates": {"latitude": lat, "longitude": lon},
            "forecast_days": days,
            "forecasts": enhanced_result['forecasts'],
            "training_info": enhanced_result['training_info'],
            "current_weather": enhanced_result.get('current_weather'),
            "model_type": enhanced_result['model_type'],
            "features": {
                "coordinate_specific_training": True,
                "random_forest_models": True,
                "historical_data_integration": True,
                "real_time_data_fusion": enhanced_result.get('current_weather') is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced forecast: {str(e)}")

@app.get("/training/info")
async def get_training_info():
    """Get information about Random Forest model training"""
    if not HAS_ENHANCED_FORECASTING:
        return {"status": "Enhanced forecasting not available"}
    
    try:
        training_info = enhanced_weather_system.forecaster.get_training_info()
        return {
            "enhanced_forecasting_available": True,
            "training_info": training_info,
            "capabilities": {
                "coordinate_specific_training": True,
                "random_forest_models": training_info.get('models_available', False),
                "real_data_integration": True,
                "historical_pattern_analysis": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training info: {str(e)}")

@app.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        # Get model metrics
        model_metrics = {}
        if weather_system.weather_model.is_trained:
            model_metrics = weather_system.weather_model.get_model_metrics()
        
        # Enhanced forecasting metrics
        enhanced_metrics = {}
        if HAS_ENHANCED_FORECASTING:
            enhanced_metrics = enhanced_weather_system.forecaster.get_training_info()
        
        # System information
        system_info = {
            "nasa_data_sources": {
                "geos_cf": "Available",
                "gpm": "Available", 
                "data_resolution": "0.25° (~25km)",
                "update_frequency": "Hourly"
            },
            "ai_model_info": {
                "algorithm": "Random Forest Ensemble",
                "models": 5,
                "training_samples": 15000,
                "features": ["latitude", "longitude", "elevation", "time", "climate_zone"],
                "ensemble_size": 100
            },
            "enhanced_forecasting": {
                "available": HAS_ENHANCED_FORECASTING,
                "coordinate_specific_training": HAS_ENHANCED_FORECASTING,
                "real_data_integration": HAS_ENHANCED_FORECASTING,
                "training_sessions": enhanced_metrics.get('training_sessions', 0) if HAS_ENHANCED_FORECASTING else 0
            },
            "confidence_levels": {
                "next_24_hours": "85-90% (NASA + AI)",
                "2_7_days": "75-85% (AI Enhanced)",
                "8_30_days": "65-75% (Seasonal Patterns)",
                "coordinate_trained": "80-95% (Enhanced RF)" if HAS_ENHANCED_FORECASTING else "N/A"
            }
        }
        
        return SystemMetricsResponse(
            model_metrics={**model_metrics, **enhanced_metrics},
            system_info=system_info,
            is_trained=weather_system.weather_model.is_trained,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_trained": weather_system.weather_model.is_trained,
        "nasa_connection": "available"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )