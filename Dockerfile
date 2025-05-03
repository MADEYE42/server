# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV MODEL_PATH=model_path.pth
ENV FLASK_APP=app.py

# Expose port for Cloud Run or external access
EXPOSE 8080

# Start the app using Gunicorn with WSGI entry point
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app"]
