# Drift-ML Application with Security Components
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for data and configuration
RUN mkdir -p /app/data /app/config /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SECURITY_DB_URL=sqlite:///data/security.db
ENV JWT_SECRET=your-jwt-secret-key
ENV LOG_LEVEL=INFO

# Expose ports for APIs
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main"]