@echo off
cd /d "%~dp0backend"
set PYTHONPATH=%~dp0
echo ============================================================
echo Starting Sentiment Analysis Backend Server v2.0
echo ============================================================
py -3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
