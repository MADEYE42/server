from flask import Flask, request, jsonify, send_from_directory
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
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains with proper configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# Directory for file uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variables
model = None
model_status = {
    "loaded": False,
    "loading": False,
    "error": None
}

def load_model_on_startup():
    global model, model_status
    if model_status["loading"]:
        return
        
    model_status["loading"] = True
    try:
        logger.info("Starting model loading process")
        MODEL_PATH = "model_path.pth"
        # Force CPU usage to avoid CUDA issues on Render
        device = torch.device("cpu")
        logger.info(f"Loading model on device: {device}")
        
        # Add a reduced timeout for Render environment
        model = load_model(MODEL_PATH, num_classes=10, device=device)
        
        if model is not None:
            model_status["loaded"] = True
            model_status["error"] = None
            logger.info("Model loaded successfully")
        else:
            model_status["error"] = "Model failed to load but no exception was thrown"
            logger.error(model_status["error"])
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        model_status["error"] = str(e)
    finally:
        model_status["loading"] = False

# Add health endpoint that doesn't depend on model loading
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "model_loaded": model_status["loaded"],
        "model_loading": model_status["loading"],
        "model_error": model_status["error"]
    })

# Start model loading in a separate thread
@app.route('/load-model', methods=['POST'])
def trigger_model_load():
    if not model_status["loaded"] and not model_status["loading"]:
        threading.Thread(target=load_model_on_startup, daemon=True).start()
        return jsonify({"message": "Model loading started"})
    elif model_status["loading"]:
        return jsonify({"message": "Model is already loading"})
    elif model_status["loaded"]:
        return jsonify({"message": "Model is already loaded"})

# Route for uploading image and JSON files
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_files():
    start_time = time()
    
    # Special handling for OPTIONS requests
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight request successful"})
        return response
        
    try:
        # Check if model is loaded
        global model, model_status
        if not model_status["loaded"]:
            # If model isn't loaded yet and not currently loading, start loading it
            if not model_status["loading"]:
                threading.Thread(target=load_model_on_startup, daemon=True).start()
                
            return jsonify({
                'error': 'Model is not loaded yet',
                'model_status': model_status
            }), 503
        
        # Check if files are included
        if 'image' not in request.files or 'json' not in request.files:
            return jsonify({'error': 'No image or JSON file part in the request'}), 400
        
        image_file = request.files['image']
        json_file = request.files['json']
        
        if image_file.filename == '' or json_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

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
        
        logger.info(f"Files saved: {image_path}, {json_path}")

        # Load the JSON and Image
        data, image = load_json_and_image(json_path, image_path)
        if data is None or image is None:
            return jsonify({"error": "Failed to load files"}), 400

        # Perform Segmentation
        segmented_image = draw_segmentation(data, image)
        if segmented_image is None:
            return jsonify({"error": "Segmentation failed"}), 500

        # Save the segmented image with unique name
        segmented_image_filename = f"segmented_{unique_id}.jpg"
        segmented_image_path = os.path.join(RESULTS_FOLDER, segmented_image_filename)
        cv2.imwrite(segmented_image_path, segmented_image)
        
        logger.info(f"Segmented image saved at: {segmented_image_path}")

        # Perform Prediction
        try:
            classes = ["3VT", "ARSA", "AVSD", "Dilated Cardiac Sinus", "ECIF", "HLHS", "LVOT", "Normal Heart", "TGA", "VSD"]
            predictions = predict_single_image(segmented_image_path, model, classes, torch.device("cpu"))
            
            if predictions is None:
                return jsonify({"error": "Prediction failed"}), 500
                
            logger.info(f"Predictions: {predictions}")
            
            # Return results to frontend
            return jsonify({
                "predictions": predictions,
                "annotations": data["shapes"],
                "segmented_image": f'/results/{segmented_image_filename}'
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error in upload process: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# Route to serve the segmented images
@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

# Route to serve uploaded files (if needed)
@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Catch-all route for undefined endpoints
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({
        "message": "Welcome to Heart Disease Detection API",
        "status": "ok",
        "model_loaded": model_status["loaded"],
        "model_loading": model_status["loading"],
        "model_error": model_status["error"],
        "documentation": "Please use /upload endpoint for predictions"
    })

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Start model loading in background
    threading.Thread(target=load_model_on_startup, daemon=True).start()
    # Debug set to False for production
    app.run(host="0.0.0.0", port=port, debug=False)