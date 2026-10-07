# Dockerfile

# Use a lightweight official Python image.
FROM python:3.12-slim

# Prevent Python from creating .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately in Docker logs.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Copy API runtime dependencies first.
# This allows Docker to cache the dependency-installation layer.
COPY requirements-api.txt .

# Install runtime dependencies.
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the application code.
COPY api/ ./api/
COPY src/ ./src/

# Copy the inference model artifacts.
COPY artifacts/models/ ./artifacts/models/

# Expose the port used by the FastAPI application.
EXPOSE 8000

# Start the FastAPI application with Uvicorn.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]