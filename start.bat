@echo off
echo Customer Churn Prediction - Quick Start
echo =====================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

echo.
echo Checking for data files...
if not exist "data\train.csv" (
    echo WARNING: Training data not found
    echo Please download train.csv and test.csv from Kaggle and place them in the data\ folder
    echo Competition: https://www.kaggle.com/competitions/binaryclassificationwithabankchurndatasetumgc
    echo.
)

echo.
echo Starting backend server...
start "Backend API" cmd /k "cd backend && python main.py"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting frontend (if Node.js is available)...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    cd frontend
    npm install
    start "Frontend" cmd /k "npm start"
    cd ..
) else (
    echo Node.js not found. Frontend will not start.
    echo Please install Node.js from https://nodejs.org to run the frontend.
)

echo.
echo =====================================
echo Setup complete!
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000 (if Node.js is installed)
echo.
echo Press any key to exit...
pause >nul
