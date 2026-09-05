@echo off
title LoanPredict AI - Explainable AI Platform
echo ============================================================
echo   Starting LoanPredict AI - Explainable AI Platform
echo ============================================================
if exist .venv\Scripts\python.exe (
    echo Using Virtual Environment (.venv)...
    .venv\Scripts\python.exe app.py
) else (
    echo Using System Python...
    python app.py
)
pause
