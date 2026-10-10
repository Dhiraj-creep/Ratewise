@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
if not exist .venv\ratewise-ready (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
  echo installed>.venv\ratewise-ready
)
.venv\Scripts\python.exe -m streamlit run app.py
goto :eof
:failed
echo Setup failed. Install Python 3.12 and check your internet connection, then retry.
pause
