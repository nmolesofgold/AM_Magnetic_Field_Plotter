@echo off
title Magnetic Field Analyzer
echo Starting Magnetic Plotting App...
echo Path: D:\PAGRAS Project\1 Project Management\Data\Magnetic Plotting Software\

:: Switch to the correct drive and folder
cd /d "D:\PAGRAS Project\1 Project Management\Data\Magnetic Plotting Software"

:: Run the Streamlit app
python -m streamlit run Magnetic_plot_app.py

:: Keep the window open if there is an error
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Please check the error message above.
    pause
)