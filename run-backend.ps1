# Backend Server Runner
# Run this script directly: .\run-backend.ps1
# Or via make: make start-backend

Write-Host "Starting FastAPI backend server..." -ForegroundColor Yellow
Write-Host "Listen on: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API Docs:  http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Set-Location backend
C:/Python313/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
