FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (required for technical indicators)
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib

# Copy requirements first for better caching
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/models reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "launch_ml_trading.py", "--mode", "trade", "--paper"]