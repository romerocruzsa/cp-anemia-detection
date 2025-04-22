# Base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]