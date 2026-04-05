FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install dependencies directly from pyproject.toml requirements
RUN pip install --no-cache-dir curl-cffi orjson pydantic fastapi uvicorn loguru

# Set PYTHONPATH to include src directory
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose the port
EXPOSE 8000

# Start the FastAPI server using the absolute path to the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
