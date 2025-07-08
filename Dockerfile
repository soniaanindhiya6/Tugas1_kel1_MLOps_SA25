# Base Python image
FROM python:3.10-slim-bullseye

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Default command runs training
CMD ["python", "train.py"]
