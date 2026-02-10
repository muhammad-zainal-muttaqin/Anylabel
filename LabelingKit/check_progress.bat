@echo off
echo.
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo [ERROR] Virtual environment not found
    pause
    exit /b 1
)

python scripts\verify_labels.py
echo.
python scripts\label_stats.py

pause
