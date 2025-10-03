# NASA Weather Prediction API - PowerShell Setup Script

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   NASA Weather Prediction API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check pip availability
try {
    $pipVersion = pip --version 2>&1
    Write-Host "Found pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: pip is not available" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 1: Installing core dependencies..." -ForegroundColor Yellow

# Install minimal requirements first
try {
    pip install -r requirements-minimal.txt
    Write-Host "✅ Core dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Trying alternative installation method..." -ForegroundColor Yellow
    try {
        pip install fastapi uvicorn pydantic requests python-multipart
        Write-Host "✅ Core dependencies installed via alternative method" -ForegroundColor Green
    } catch {
        Write-Host "❌ Core installation failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "Step 2: Installing scientific computing packages..." -ForegroundColor Yellow
Write-Host "(This may take a while and some packages might fail - that's OK)" -ForegroundColor Gray
Write-Host ""

# Function to install package with error handling
function Install-Package {
    param([string]$PackageName, [string]$PackageSpec)
    
    Write-Host "Installing $PackageName..." -ForegroundColor Cyan
    try {
        $output = pip install $PackageSpec 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $PackageName installed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "⚠️ $PackageName installation failed" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "⚠️ $PackageName installation failed" -ForegroundColor Yellow
        return $false
    }
}

# Try to install scientific packages
$packages = @{
    "numpy" = "numpy>=1.24.0,<2.0.0"
    "pandas" = "pandas>=2.0.0" 
    "scikit-learn" = "scikit-learn>=1.3.0"
    "scipy" = "scipy>=1.10.0"
    "matplotlib" = "matplotlib>=3.7.0"
    "seaborn" = "seaborn>=0.12.0"
    "xarray" = "xarray>=2023.1.0"
}

$successCount = 0
$totalPackages = $packages.Count

foreach ($package in $packages.GetEnumerator()) {
    if (Install-Package -PackageName $package.Key -PackageSpec $package.Value) {
        $successCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Summary:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Core packages: Installed" -ForegroundColor Green
Write-Host "📊 Scientific packages: $successCount/$totalPackages installed" -ForegroundColor $(if ($successCount -eq $totalPackages) { "Green" } else { "Yellow" })

if ($successCount -lt $totalPackages) {
    Write-Host ""
    Write-Host "Note: Some scientific packages failed to install." -ForegroundColor Yellow
    Write-Host "The API will automatically use simplified models with reduced functionality." -ForegroundColor Yellow
    Write-Host "This is normal for some Python distributions or older systems." -ForegroundColor Gray
}

Write-Host ""
Write-Host "Starting NASA Weather Prediction API..." -ForegroundColor Green
Write-Host ""
Write-Host "The API will be available at:" -ForegroundColor Cyan
Write-Host "  - http://localhost:8000" -ForegroundColor White
Write-Host "  - http://localhost:8000/docs (API Documentation)" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start the server
try {
    python run.py
} catch {
    Write-Host "Error starting the server: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}