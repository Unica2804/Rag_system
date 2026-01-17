# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install system build tools (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment in /opt/venv
RUN python -m venv /opt/venv

# Activate the venv for the builder stage
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# 1. Upgrade pip to ensure it handles wheels correctly
# 2. Install CPU-only torch to save space
# 3. Install your requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy the entire virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Enable the virtual environment in the final image
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code
COPY . .

EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]