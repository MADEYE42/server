# Use Python 3.10 slim as the base image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Copy all source code, including model file
COPY . .

# Set environment variables
ENV MODEL_PATH=model_path.pth
ENV FLASK_APP=app.py

# Debugging: Check contents of /app
RUN ls -l /app

# Expose port for Cloud Run
EXPOSE 8080

# Start app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app"]
