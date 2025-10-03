# Troubleshooting Guide

## Common Installation Issues

### Python 3.13 Compatibility Issues

If you're encountering errors with Python 3.13, try these solutions:

#### Solution 1: Install Minimal Version First
```bash
# Install only the core FastAPI components first
pip install -r requirements-minimal.txt

# Then try installing scientific packages one by one
pip install numpy
pip install pandas  
pip install scikit-learn
```

#### Solution 2: Use Python 3.11 or 3.12
Python 3.13 is very new and some packages may not have compatible wheels yet.
- Download Python 3.11 or 3.12 from https://python.org
- Create a virtual environment with the older Python version

#### Solution 3: Use the Simplified Models
The API includes fallback models that work without heavy scientific dependencies:
```bash
# Install only the minimal requirements
pip install fastapi uvicorn pydantic requests python-multipart

# The API will automatically use simplified models
python run.py
```

### Installation Script Options

#### Windows Batch File
```bash
start.bat
```
This will try installing packages step by step and continue even if some fail.

#### PowerShell Script (Recommended for Windows)
```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```
Better error handling and progress reporting.

#### Manual Installation
```bash
# Step 1: Core packages (required)
pip install fastapi>=0.104.0
pip install uvicorn[standard]>=0.24.0  
pip install pydantic>=2.4.0
pip install requests>=2.31.0
pip install python-multipart>=0.0.6

# Step 2: Scientific packages (optional, for advanced features)
pip install numpy>=1.24.0,<2.0.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0

# Step 3: Additional packages (optional)
pip install scipy matplotlib seaborn xarray
```

### Common Error Messages

#### "Cannot import 'setuptools.build_meta'"
This is usually a Python 3.13 compatibility issue.
- **Solution**: Use Python 3.11 or 3.12, or install only minimal requirements

#### "Microsoft Visual C++ 14.0 is required"
Some packages need compilation on Windows.
- **Solution**: Install Microsoft C++ Build Tools or use pre-compiled wheels
- **Alternative**: Use the simplified models (no compilation needed)

#### "No module named 'numpy'"
Scientific packages failed to install.
- **Solution**: The API will automatically use simplified models
- **Alternative**: Try installing numpy separately: `pip install numpy`

#### Import errors in main.py
- **Solution**: The main.py file now has fallback imports for simplified models

### API Functionality Levels

#### Full Functionality (All packages installed)
- ✅ Advanced Random Forest models
- ✅ NASA data integration
- ✅ Complex statistical calculations
- ✅ All API endpoints

#### Reduced Functionality (Minimal packages only)
- ✅ Basic weather prediction models
- ✅ Simple statistical calculations  
- ✅ All API endpoints
- ⚠️ Reduced prediction accuracy
- ⚠️ No NASA data integration

#### Core Functionality (FastAPI only)
- ✅ API server runs
- ✅ Basic endpoints respond
- ⚠️ Weather predictions use fallback algorithms

### Testing Your Installation

1. **Test the API server**:
   ```bash
   python run.py
   ```
   Should start without errors.

2. **Test API endpoints**:
   ```bash
   python test_api.py
   ```
   Or visit http://localhost:8000/docs

3. **Check what features are available**:
   Visit http://localhost:8000/metrics to see which models are active.

### Performance Notes

- **Simplified models** are faster but less accurate
- **Full models** provide better predictions but require more resources
- **NASA integration** provides real-time data but requires internet connectivity

### Getting Help

If you continue to have issues:

1. Check the console output for specific error messages
2. Try the minimal installation first
3. Use Python 3.11 or 3.12 instead of 3.13
4. Check if you're behind a corporate firewall (affects package downloads)
5. Consider using Docker if local installation is problematic

### Docker Alternative

If local installation is too problematic:

```bash
# Build and run with Docker
docker-compose up --build
```

This avoids all local Python dependency issues.