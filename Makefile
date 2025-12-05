.PHONY: install install-backend install-frontend start-backend start-frontend start dev clean help

help:
	@echo "Available commands:"
	@echo "  make install-backend - Install backend dependencies"
	@echo "  make install-frontend- Install frontend dependencies"
	@echo "  make install         - Install dependencies for both backend and frontend"
	@echo "  make start-backend   - Start the backend server"
	@echo "  make start-frontend  - Start the frontend development server"
	@echo "  make start          - Start both backend and frontend"
	@echo "  make dev            - Start development environment"
	@echo "  make clean          - Clean up dependencies and cache files"

install-backend:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

install: install-backend install-frontend

start-backend:
	@echo "Starting backend server..."
	cd backend && python main.py

start-frontend:
	@echo "Starting frontend development server..."
	cd frontend && npm run dev

start: start-backend start-frontend

dev: start

clean:
	@echo "Cleaning backend cache..."
	cd backend && rm -rf __pycache__ *.pyc
	@echo "Cleaning frontend cache..."
	cd frontend && rm -rf .next node_modules
	@echo "Clean complete!"
