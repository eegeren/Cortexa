# Railway-optimized Dockerfile for Cortexa
FROM python:3.11

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY requirements.txt .
COPY frontend/package*.json ./frontend/

# Install dependencies
RUN pip install -r requirements.txt --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/
RUN cd frontend && npm install

# Copy all files
COPY . .

# Build frontend
RUN cd frontend && npm run build && cp -r build ../static

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]