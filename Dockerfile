# Ultra Simple Railway Dockerfile - Backend Only
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

# Copy main application
COPY main.py .

# Create empty static directory for Railway
RUN mkdir -p static

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]