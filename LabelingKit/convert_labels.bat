@echo off
echo ============================================
echo   Convert JSON Labels to YOLO Format
echo ============================================
echo.

call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo [ERROR] Virtual environment not found
    echo         Run setup.bat first
    pause
    exit /b 1
)

python scripts\convert_json_to_yolo.py

echo.
pause
