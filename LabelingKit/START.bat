@echo off
echo.
echo Membuka AnyLabeling...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Belum diinstall!
    echo         Jalankan INSTALL.bat terlebih dahulu.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
start "" anylabeling
exit
