@echo off
cd /d "%~dp0absa\absa"
echo ===================================================
echo   Menjalankan Aplikasi Web ABSA...
echo   Silakan buka browser di: http://127.0.0.1:5000
echo   Untuk berhenti, tekan Ctrl + C
echo ===================================================
echo.
.\.venv\Scripts\python.exe app.py
pause
