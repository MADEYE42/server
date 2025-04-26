from flask import Flask, request, jsonify, send_from_directory
import os
import json
import cv2
import numpy as np
import torch
import logging
from time import time
import threading
import traceback
import sys
import gc

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Force model path to be absolute
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model_path.pth")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains with proper configuration
# Import the flask_cors only after confirming it's installed
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
    logger.info("CORS initialized successfully")
except ImportError:
    logger.error("flask_cors not installed. CORS support will not be available.")
except Exception as e:
    logger.error(f"Error initializing CORS: {str(e)}")

# Directory for file uploads and results
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(CURRENT_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
logger.info(f"Created directories: {UPLOAD_FOLDER}, {RESULTS_FOLDER}")

# Global variables
model = None
model_status = {
    "loaded": False,
    "loading": False,
    "error": None,
    "last_attempt": None
}

# Import prediction functions only when needed to avoid startup failures
def import_prediction_modules():
    try:
        logger.info("Importing prediction and segmentation modules...")
        global load_json_and_image, draw_segmentation, predict_single_image, load_model
        
        # First, try to import from the current directory
        sys.path.insert(0, CURRENT_DIR)
        
        from segmentation import load_json_and_image, draw_segmentation
        from prediction import predict_single_image, load_model
        
        logger.info("Successfully imported prediction and segmentation modules")
        return True
    except Exception as e:
        logger.error(f"Failed to import modules: {str(e)}\n{traceback.format_exc()}")
        return False

# Execute import at startup
import_success = import_prediction_modules()

def load_model_function():
    global model, model_status
    
    if model_status["loading"]:
        logger.info("Model loading already in progress, skipping duplicate load")
        return
        
    logger.info("=== STARTING MODEL LOADING PROCESS ===")
    model_status["loading"] = True
    model_status["last_attempt"] = time()
    
    try:
        # Check if we can find the model file
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")
            
        logger.info(f"Model file exists at {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB")
        
        # Force garbage collection before loading model
        gc.collect()
        
        # Log memory usage before loading
        import psutil
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage before loading model: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Force CPU usage to avoid CUDA issues on Render
        device = torch.device("cpu")
        logger.info(f"Loading model on device: {device}")
        
        # Load the model
        model = load_model(MODEL_PATH, num_classes=10, device=device)
        
        # Check if model loaded successfully
        if model is not None:
            model_status["loaded"] = True
            model_status["error"] = None
            logger.info("Model loaded successfully!")
            
            # Log memory after loading
            logger.info(f"Memory usage after loading model: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        else:
            model_status["error"] = "Model loading returned None without raising an exception"
            logger.error(model_status["error"])
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        model_status["error"] = str(e)
    finally:
        model_status["loading"] = False
        logger.info("=== MODEL LOADING PROCESS COMPLETED ===")

# Add health endpoint that doesn't depend on model loading
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "model_loaded": model_status["loaded"],
        "model_loading": model_status["loading"],
        "model_error": model_status["error"],
        "model_path_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False,
        "import_modules_success": import_success,
        "last_load_attempt": model_status["last_attempt"]
    })

# Start model loading in a separate thread
@app.route('/load-model', methods=['GET', 'POST'])
def trigger_model_load():
    if not import_success:
        return jsonify({
            "error": "Cannot load model because required modules failed to import",
            "status": "failed"
        }), 500
        
    if not model_status["loaded"] and not model_status["loading"]:
        threading.Thread(target=load_model_function, daemon=True).start()
        return jsonify({"message": "Model loading started", "status": "loading"})
    elif model_status["loading"]:
        return jsonify({"message": "Model is already loading", "status": "loading"})
    elif model_status["loaded"]:
        return jsonify({"message": "Model is already loaded", "status": "loaded"})

# Route for uploading image and JSON files
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_files():
    start_time = time()
    
    # Special handling for OPTIONS requests
    if request.method == 'OPTIONS':
        response = jsonify({"message": "CORS preflight request successful"})
        return response
        
    if not import_success:
        return jsonify({
            "error": "Required modules failed to import. Server cannot process uploads.",
            "status": "configuration_error"
        }), 500
        
    try:
        # Check if model is loaded
        global model, model_status
        if not model_status["loaded"]:
            # If model isn't loaded yet and not currently loading, start loading it
            if not model_status["loading"]:
                threading.Thread(target=load_model_function, daemon=True).start()
                
            return jsonify({
                'error': 'Model is not loaded yet. Please try again in a few moments.',
                'model_status': model_status
            }), 503
        
        # Log the request content type and form data keys
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request form keys: {list(request.form.keys()) if request.form else None}")
        logger.info(f"Request files keys: {list(request.files.keys()) if request.files else None}")
        
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
    # Trigger model loading on first visit if not already loading
    if not model_status["loaded"] and not model_status["loading"]:
        logger.info("First visit detected, starting model loading in background")
        threading.Thread(target=load_model_function, daemon=True).start()
    
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
    
    # Trigger model loading on startup
    logger.info("Application starting, triggering model loading")
    threading.Thread(target=load_model_function, daemon=True).start()
    
    # Debug set to False for production
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)