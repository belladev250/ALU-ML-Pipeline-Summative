FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p uploads visualizations

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/api.py

# Run API
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]