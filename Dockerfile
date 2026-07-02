# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment system variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENV=production

# Set system work directory
WORKDIR /app

# Install system build dependencies required for compiling libraries (e.g., FAISS compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker build cache layers
COPY requirements.txt .

# Install python package requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source directories and dataset files
COPY src/ ./src/
COPY data/ ./data/

# Expose standard API port
EXPOSE 8000

# Perform container startup verification check and start application server
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
