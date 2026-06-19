@echo off
cd /d "%~dp0"

where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

streamlit run app.py
pause
