FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install the library and its dependencies
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir fastapi uvicorn

# Expose the port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.py:app", "--host", "0.0.0.0", "--port", "8000"]
