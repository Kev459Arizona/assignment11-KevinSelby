# Use a base image with Python
FROM python:3.12-slim

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose necessary ports for the MLflow server and the Flask API
EXPOSE 5000 5001

# Define the entrypoint to run both MLflow and the Flask app
CMD ["sh", "-c", "mlflow models serve --model-uri models:/basic_lr_iris_model/1 --host 0.0.0.0 --port 5001 & flask run --host=0.0.0.0 --port 5000"]
