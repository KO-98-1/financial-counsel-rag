@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
echo Starting the counseling RAG app. A browser tab opens in a few seconds. Press Ctrl+C here to stop.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run src\counsel\app.py --server.port 8501
pause
