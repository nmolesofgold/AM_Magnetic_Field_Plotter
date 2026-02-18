@echo off
title Magnetic Analyzer Setup
echo ====================================================
echo   INSTALLING LIBRARIES FOR MAGNETIC PLOTTER
echo ====================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python from python.org before running this setup.
    pause
    exit
)

echo [1/2] Upgrading Pip...
python -m pip install --upgrade pip

echo [2/2] Installing required libraries from requirements.txt...
pip install -r requirements.txt

echo.
echo ====================================================
echo   SETUP COMPLETE! 
echo   You can now use your Launch_Magnet_App.bat file.
echo ====================================================
pause