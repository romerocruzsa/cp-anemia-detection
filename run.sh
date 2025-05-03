#!/bin/bash
# CONFIG/local.env must define AES_KEY

echo "Starting FastAPI backend on port 8000..."
cd backend
echo "Loading environment variables..."
set -o allexport
source CONFIG/local.env
set +o allexport
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
cd ..

echo "Serving frontend at http://localhost:5500"
python3 -m http.server 5500
