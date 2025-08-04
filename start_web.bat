@echo off
echo ====================================================
echo   CA DAO - TUC NGU SEARCH ENGINE
echo   Powered by RankNet Algorithm
echo ====================================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)
echo ✓ Python found

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

echo.
echo [3/3] Starting the web application...
echo.
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
pause

python app.py
