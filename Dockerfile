FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install dependencies including database libraries
RUN pip install --no-cache-dir \
    curl-cffi \
    orjson \
    pydantic \
    fastapi \
    uvicorn \
    loguru \
    sqlalchemy \
    psycopg2-binary

# Set PYTHONPATH to include src directory
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose the port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
