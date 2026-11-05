@echo off
REM ThermalForge Studio - double-click to start the demo.
cd /d "%~dp0"
python run_studio.py %*
if errorlevel 1 pause
