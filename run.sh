#!/bin/bash

# Run FastAPI backend
echo "Starting backend..."
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
cd ..

# Run simple Python HTTP server for frontend
echo "Serving frontend at http://localhost:5500"
python3 -m http.server 5500