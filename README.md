# NASA Weather Prediction API

A FastAPI-based backend for global weather prediction using NASA data and AI models.

## Features

- 🛰️ **NASA Data Integration**: Real-time weather data from NASA GEOS-CF and GPM satellites  
- 🤖 **AI Weather Prediction**: Random Forest models for accurate weather forecasting
- 🌍 **Global Coverage**: Weather predictions for any location worldwide
- 📅 **Multi-day Forecasts**: Up to 30-day weather forecasts
- 🎯 **Travel Recommendations**: Smart recommendations based on weather conditions
- 📊 **Performance Metrics**: Model accuracy and system performance monitoring

## API Endpoints

### Weather Data
- `GET /weather/coordinates` - Get weather by latitude/longitude
- `POST /weather/location` - Get weather by location name
- `GET /weather/future` - Get weather prediction for specific future date

### Forecasts
- `GET /forecast/coordinates` - Get multi-day forecast by coordinates (with enhanced RF option)
- `POST /forecast/location` - Get multi-day forecast by location name
- `GET /forecast/enhanced` - Get enhanced Random Forest forecast with coordinate-specific training

### System
- `GET /metrics` - System performance metrics
- `GET /training/info` - Random Forest training information
- `GET /health` - Health check endpoint
- `GET /` - API information

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   python run.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Get Weather by Coordinates
```bash
curl "http://localhost:8000/weather/coordinates?lat=40.7128&lon=-74.0060"
```

### Get Weather by Location Name
```bash
curl -X POST "http://localhost:8000/weather/location" \
  -H "Content-Type: application/json" \
  -d '{"location": "New York, USA", "source": "nasa"}'
```

### Get 7-Day Forecast
```bash
curl "http://localhost:8000/forecast/coordinates?lat=40.7128&lon=-74.0060&days=7"
```

### Get Enhanced Random Forest Forecast
```bash
# Enhanced forecast with coordinate-specific training
curl "http://localhost:8000/forecast/enhanced?lat=40.7128&lon=-74.0060&days=7"

# With OpenWeatherMap API key for real data integration
curl "http://localhost:8000/forecast/enhanced?lat=40.7128&lon=-74.0060&days=7&openweather_key=YOUR_API_KEY"
```

### Get Future Weather Prediction
```bash
curl "http://localhost:8000/weather/future?lat=40.7128&lon=-74.0060&date=2024-12-25"
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Data Sources

### NASA GEOS Composition Forecast (GEOS-CF)
- Global weather and atmospheric composition forecasting
- Resolution: 0.25° latitude/longitude (~25km)
- Parameters: Temperature, Humidity, Wind, Pressure
- Update frequency: Hourly

### NASA Global Precipitation Measurement (GPM)
- Multi-satellite precipitation measurements
- Resolution: 0.1° latitude/longitude (~10km)
- Coverage: Global precipitation data
- Update frequency: Daily

### AI Models
- **Algorithm**: Random Forest Ensemble
- **Training Data**: 15,000+ synthetic weather patterns
- **Models**: 5 specialized predictors (temperature, humidity, precipitation, wind, conditions)
- **Features**: Geographic coordinates, elevation, temporal factors, climate zones

## Response Format

### Weather Response
```json
{
  "location": "New York, New York, USA",
  "coordinates": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "elevation": 0
  },
  "weather": {
    "temperature": 22.5,
    "humidity": 65.2,
    "wind_speed": 4.8,
    "precipitation_probability": 25.0,
    "weather_condition": "pleasant",
    "pressure": 1013.25,
    "source": "NASA GEOS-CF & GPM",
    "confidence": 88,
    "nasa_data_used": true
  },
  "probabilities": {
    "hot": 15,
    "warm": 85,
    "pleasant": 90
  },
  "travel_recommendation": {
    "rating": "👍 GOOD",
    "score": 78,
    "recommendation": "Good weather conditions. Suitable for most activities.",
    "details": {
      "temperature_suitability": "Ideal",
      "precipitation_outlook": "Low chance",
      "wind_conditions": "Calm"
    }
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Location not found
- `500`: Internal server error

Error responses include detailed messages:
```json
{
  "detail": "Location not found",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Configuration

Edit `config.py` to customize:
- Server settings (host, port, debug mode)
- NASA data URLs
- Model parameters
- Forecast limits
- CORS settings

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI application
├── run.py               # Server startup script
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── models/
│   ├── __init__.py
│   └── weather_models.py # Weather prediction models
└── schemas/
    ├── __init__.py
    └── weather_schemas.py # Pydantic schemas
```

### Adding New Features

1. **New Endpoints**: Add to `main.py`
2. **Data Models**: Update `schemas/weather_schemas.py`
3. **Weather Logic**: Modify `models/weather_models.py`
4. **Configuration**: Update `config.py`

## License

This project is part of the NASA Space Apps Challenge submission.

## Support

For issues and questions, please refer to the API documentation at `/docs` or `/redoc` endpoints.