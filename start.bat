@echo off
echo.
echo ========================================
echo   NASA Weather Prediction API
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo.
echo Step 1: Installing core dependencies...
pip install -r requirements-minimal.txt

if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    echo Trying alternative installation method...
    pip install fastapi uvicorn pydantic requests python-multipart
    if errorlevel 1 (
        echo ERROR: Core installation failed
        pause
        exit /b 1
    )
)

echo.
echo Step 2: Attempting to install scientific computing packages...
echo (This may take a while and some packages might fail - that's OK)
echo.

REM Try to install scientific packages one by one
echo Installing numpy...
pip install "numpy>=1.24.0,<2.0.0" >nul 2>&1
if errorlevel 1 (
    echo Warning: numpy installation failed, using fallback math functions
) else (
    echo Success: numpy installed
)

echo Installing pandas...
pip install "pandas>=2.0.0" >nul 2>&1
if errorlevel 1 (
    echo Warning: pandas installation failed, using basic data structures
) else (
    echo Success: pandas installed
)

echo Installing scikit-learn...
pip install "scikit-learn>=1.3.0" >nul 2>&1
if errorlevel 1 (
    echo Warning: scikit-learn installation failed, using simple prediction models
) else (
    echo Success: scikit-learn installed
)

echo Installing remaining scientific packages...
pip install scipy matplotlib seaborn xarray >nul 2>&1
if errorlevel 1 (
    echo Warning: Some scientific packages failed to install
    echo The API will work with reduced functionality
) else (
    echo Success: Scientific packages installed
)

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.
echo Starting NASA Weather Prediction API...
echo.
echo The API will be available at:
echo   - http://localhost:8000
echo   - http://localhost:8000/docs (API Documentation)
echo.
echo Note: The API will automatically use simplified models
echo if scientific computing packages are not available.
echo.
echo Press Ctrl+C to stop the server
echo.

python run.py

pause