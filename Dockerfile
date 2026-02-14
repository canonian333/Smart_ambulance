FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Optimize pip and install CPU-only torch first to save space/bandwidth
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt --default-timeout=100

# Copy the rest of the application's code
COPY . .

# Expose port 8000 for the API
EXPOSE 8000

# Define environment variable
ENV PORT=8000

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
