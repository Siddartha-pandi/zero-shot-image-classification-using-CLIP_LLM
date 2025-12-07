.PHONY: install install-backend install-frontend start-backend start-frontend start dev clean help test-backend add-classes kill-backend

# Python executable path
PYTHON = C:/Python313/python.exe

help:
	@echo "Available commands:"
	@echo "  make install-backend  - Install backend dependencies"
	@echo "  make install-frontend - Install frontend dependencies"
	@echo "  make install          - Install dependencies for both backend and frontend"
	@echo "  make start-backend    - Start the FastAPI backend server"
	@echo "  make start-frontend   - Start the Next.js frontend development server"
	@echo "  make start            - Start both backend and frontend (recommended)"
	@echo "  make dev              - Start development environment (alias for start)"
	@echo "  make test-backend     - Test backend health endpoint"
	@echo "  make add-classes      - Add default classes to backend"
	@echo "  make kill-backend     - Kill all Python backend processes"
	@echo "  make clean            - Clean up dependencies and cache files"

install-backend:
	@echo "Installing backend dependencies..."
	cd backend && $(PYTHON) -m pip install --upgrade pip
	cd backend && $(PYTHON) -m pip install -r requirements.txt
	@echo "Backend dependencies installed successfully!"

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Frontend dependencies installed successfully!"

install: install-backend install-frontend
	@echo "All dependencies installed!"

start-backend:
	@echo "Starting FastAPI backend server..."
	@echo "Backend will run at http://127.0.0.1:8000"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd backend && $(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

start-frontend:
	@echo "Starting Next.js frontend development server..."
	@echo "Frontend will run at http://localhost:3000"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd frontend && npm run dev

start:
	@echo "========================================="
	@echo "Starting Development Environment"
	@echo "========================================="
	@echo ""
	@echo "Backend:  http://127.0.0.1:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API Docs: http://127.0.0.1:8000/docs"
	@echo ""
	@echo "Note: Start backend and frontend in separate terminals:"
	@echo "  Terminal 1: make start-backend"
	@echo "  Terminal 2: make start-frontend"
	@echo ""
	@echo "Or run them in background (Windows):"
	@echo "  make start-backend-bg"
	@echo "  make start-frontend-bg"
	@echo "========================================="

start-backend-bg:
	@echo "Starting backend in new window..."
	@powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd ''S:\Siddu\Final Year\zero-shot''; make start-backend'"
	@echo "Backend started in new window!"

start-frontend-bg:
	@echo "Starting frontend in new window..."
	@powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd ''S:\Siddu\Final Year\zero-shot''; make start-frontend'"
	@echo "Frontend started in new window!"

start-all-bg: start-backend-bg
	@powershell -Command "Start-Sleep -Seconds 2"
	@$(MAKE) start-frontend-bg
	@echo ""
	@echo "Both servers started in separate windows!"
	@echo "Backend:  http://127.0.0.1:8000"
	@echo "Frontend: http://localhost:3000"

dev: start

check-backend:
	@echo "Checking if backend is running..."
	@powershell -Command "try { $$r = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; Write-Host 'Backend is RUNNING - Status: '$$r.status -ForegroundColor Green; $$classes = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/classes'; Write-Host 'Classes defined: '$$classes.classes.Count -ForegroundColor Cyan } catch { Write-Host 'Backend is NOT running' -ForegroundColor Red }"

test-backend:
	@echo "Testing backend health endpoint..."
	@powershell -Command "try { $$r = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'; Write-Host 'Status: '$$r.status -ForegroundColor Green } catch { Write-Host 'Error: Backend not running or unreachable' -ForegroundColor Red }"

add-classes:
	@echo "Adding default classes to backend..."
	cd backend && $(PYTHON) test_add_classes.py

kill-backend:
	@echo "Killing backend processes..."
	@taskkill /F /IM python.exe /T 2>nul || echo "No Python processes found"
	@taskkill /F /FI "WINDOWTITLE eq *uvicorn*" 2>nul || echo "No uvicorn processes found"

clean:
	@echo "Cleaning backend cache..."
	@if exist backend\__pycache__ rd /s /q backend\__pycache__
	@if exist backend\app\__pycache__ rd /s /q backend\app\__pycache__
	@del /s /q backend\*.pyc 2>nul || echo ""
	@echo "Cleaning frontend cache..."
	@if exist frontend\.next rd /s /q frontend\.next
	@if exist frontend\node_modules rd /s /q frontend\node_modules
	@echo "Clean complete!"
