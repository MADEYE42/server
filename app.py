from flask import Flask, request, jsonify, send_from_directory, Response
import os
import json
import cv2
import numpy as np
from segmentation import load_json_and_image, draw_segmentation
from prediction import predict_single_image, load_model
from PIL import Image
import torch
from flask_cors import CORS
import logging
from time import time
import threading
import gc

# Set up logging to capture detailed errors
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains
CORS(app, supports_credentials=False)

# Directory for file uploads and results
UPLOAD_FOLDER = 'Uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variable to track if model is loaded
model_loaded = False
model = None
model_lock = threading.Lock()

def load_model_on_demand():
    global model, model_loaded
    with model_lock:
        if not model_loaded:
            try:
                # Force garbage collection before loading model
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Use environment variable with fallback
                MODEL_PATH = os.environ.get("MODEL_PATH", "model_path.pth")
                
                # Log the absolute path for debugging
                abs_path = os.path.abspath(MODEL_PATH)
                logging.info(f"Attempting to load model from {abs_path}")
                
                # Check if file exists before loading
                if not os.path.exists(MODEL_PATH):
                    logging.error(f"Model file not found at {abs_path}")
                    raise FileNotFoundError(f"Model file not found at {abs_path}")
                
                device = torch.device("cpu")
                model = load_model(MODEL_PATH, num_classes=10, device=device)
                
                if model is None:
                    raise ValueError("Model loading returned None")
                    
                model_loaded = True
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}", exc_info=True)
                model_loaded = False
                model = None
        else:
            logging.info("Model already loaded")

# Helper function to add CORS headers to all responses
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS,HEAD')
    return response

# Handle 405 Method Not Allowed errors
@app.errorhandler(405)
def method_not_allowed(error):
    logging.error(f"Method Not Allowed: {request.method} on {request.path}")
    response = jsonify({"error": f"Method {request.method} not allowed for {request.path}"})
    response.status_code = 405
    return add_cors_headers(response)

# Handle 500 Internal Server errors
@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"Server error: {str(error)}", exc_info=True)
    response = jsonify({"error": "Internal server error"})
    response.status_code = 500
    return add_cors_headers(response)

# Root endpoint for health checks
@app.route('/', methods=['GET', 'HEAD'])
def root():
    logging.info(f"Received {request.method} request to / from {request.remote_addr}")
    return jsonify({"status": "ok", "message": "Server is running", "model_loaded": model_loaded}), 200

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    logging.info(f"Received GET request to /health from {request.remote_addr}")
    return jsonify({"status": "ok", "model_loaded": model_loaded}), 200

# Debug model path and existence
@app.route('/debug-model', methods=['GET'])
def debug_model():
    MODEL_PATH = os.environ.get("MODEL_PATH", "model_path.pth")
    
    response_data = {
        "model_path": MODEL_PATH,
        "absolute_path": os.path.abspath(MODEL_PATH),
        "file_exists": os.path.exists(MODEL_PATH),
        "current_directory": os.getcwd(),
        "directory_contents": os.listdir('.')
    }
    
    if os.path.exists(MODEL_PATH):
        response_data["file_size_mb"] = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    return jsonify(response_data)

# Route to manually trigger model loading
@app.route('/load-model', methods=['GET'])
def load_model_route():
    """Route to manually trigger model loading and check status"""
    start_time = time()
    
    # Attempt to load the model
    load_model_on_demand()
    
    # Check and report status
    if model_loaded:
        response = jsonify({
            "status": "success", 
            "message": "Model loaded successfully",
            "loading_time": f"{time() - start_time:.3f} seconds"
        })
        return add_cors_headers(response)
    else:
        response = jsonify({
            "status": "error", 
            "message": "Failed to load model. Check server logs for details.",
            "loading_time": f"{time() - start_time:.3f} seconds"
        }), 500
        return add_cors_headers(response[0]), response[1]

# Favicon endpoint
@app.route('/favicon.ico', methods=['GET'])
def favicon():
    logging.info(f"Received GET request to /favicon.ico from {request.remote_addr}")
    return Response(status=204)

# Route for uploading image and JSON files
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_files():
    global model_loaded, model
    start_time = time()
    logging.debug(f"Received {request.method} request to /upload from {request.remote_addr}")
    
    # Handle preflight OPTIONS requests
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        logging.info(f"OPTIONS /upload processed in {time() - start_time:.3f} seconds")
        return add_cors_headers(response)
        
    try:
        # Load model if not already loaded (with retries)
        for attempt in range(3):
            load_model_on_demand()
            if model_loaded:
                break
            logging.warning(f"Model loading attempt {attempt + 1} failed, retrying...")
            # Sleep between retries
            import time as pytime
            pytime.sleep(2)
        
        # Check if model is loaded
        if not model_loaded:
            response = jsonify({'error': 'Failed to load model after retries. Please try again later.'}), 503
            logging.info(f"POST /upload rejected (model not loaded) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]
        
        # Check if files are included
        if 'image' not in request.files or 'json' not in request.files:
            response = jsonify({'error': 'No image or JSON file part in the request'}), 400
            logging.info(f"POST /upload failed (missing files) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]
        
        image_file = request.files['image']
        json_file = request.files['json']
        
        if image_file.filename == '' or json_file.filename == '':
            response = jsonify({'error': 'No file selected'}), 400
            logging.info(f"POST /upload failed (empty filename) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]

        # Generate unique filenames to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())
        image_filename = f"{unique_id}_{image_file.filename}"
        json_filename = f"{unique_id}_{json_file.filename}"
        
        # Save the uploaded files
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        json_path = os.path.join(UPLOAD_FOLDER, json_filename)
        
        image_file.save(image_path)
        json_file.save(json_path)
        
        logging.info(f"Files saved: {image_path}, {json_path}")

        # Load the JSON and Image
        data, image = load_json_and_image(json_path, image_path)
        if data is None or image is None:
            response = jsonify({"error": "Failed to load files"}), 400
            logging.info(f"POST /upload failed (file loading) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]

        # Perform Segmentation
        segmented_image = draw_segmentation(data, image)
        if segmented_image is None:
            response = jsonify({"error": "Segmentation failed"}), 500
            logging.info(f"POST /upload failed (segmentation) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]

        # Save the segmented image with unique name
        segmented_image_filename = f"segmented_{unique_id}.jpg"
        segmented_image_path = os.path.join(RESULTS_FOLDER, segmented_image_filename)
        cv2.imwrite(segmented_image_path, segmented_image)
        
        logging.info(f"Segmented image saved at: {segmented_image_path}")

        # Perform Prediction
        try:
            classes = ["3VT", "ARSA", "AVSD", "Dilated Cardiac Sinus", "ECIF", "HLHS", "LVOT", "Normal Heart", "TGA", "VSD"]
            predictions = predict_single_image(segmented_image_path, model, classes, torch.device("cpu"))
            
            if predictions is None:
                response = jsonify({"error": "Prediction failed"}), 500
                logging.info(f"POST /upload failed (prediction) in {time() - start_time:.3f} seconds")
                return add_cors_headers(response[0]), response[1]
                
            logging.info(f"Predictions: {predictions}")
            
            # Return results to frontend
            response = jsonify({
                "predictions": predictions,
                "annotations": data["shapes"],
                "segmented_image": f'/results/{segmented_image_filename}'
            })
            logging.info(f"POST /upload completed in {time() - start_time:.3f} seconds")
            return add_cors_headers(response)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}", exc_info=True)
            response = jsonify({"error": f"Prediction error: {str(e)}"}), 500
            logging.info(f"POST /upload failed (prediction exception) in {time() - start_time:.3f} seconds")
            return add_cors_headers(response[0]), response[1]

    except Exception as e:
        logging.error(f"Error in upload process: {str(e)}", exc_info=True)
        response = jsonify({"error": str(e)}), 500
        logging.info(f"POST /upload failed (general exception) in {time() - start_time:.3f} seconds")
        return add_cors_headers(response[0]), response[1]

# Route to serve the segmented images
@app.route('/results/<filename>')
def serve_result(filename):
    start_time = time()
    response = send_from_directory(RESULTS_FOLDER, filename)
    logging.info(f"GET /results/{filename} processed in {time() - start_time:.3f} seconds")
    return add_cors_headers(response)

# Route to serve uploaded files (if needed)
@app.route('/Uploads/<filename>')
def serve_upload(filename):
    start_time = time()
    response = send_from_directory(UPLOAD_FOLDER, filename)
    logging.info(f"GET /Uploads/{filename} processed in {time() - start_time:.3f} seconds")
    return add_cors_headers(response)

# Add OPTIONS handling for all routes to support CORS preflight requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    start_time = time()
    response = app.make_default_options_response()
    logging.info(f"OPTIONS /{path} processed in {time() - start_time:.3f} seconds")
    return add_cors_headers(response)

# Main entry point (for local testing only; Render uses Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)