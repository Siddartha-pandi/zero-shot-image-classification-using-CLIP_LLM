# Frontend Server Runner
# Run this script directly: .\run-frontend.ps1
# Or via make: make start-frontend

Write-Host "Starting Next.js frontend server..." -ForegroundColor Yellow
Write-Host "Listen on: http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Set-Location frontend
npm run dev
