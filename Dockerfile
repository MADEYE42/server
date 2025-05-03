# Use a base image with Python 3.13
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port for Cloud Run (default is 8080)
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app"]
