from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
import logging
import cv2
import torch
from PIL import Image
from time import time
from segmentation import load_json_and_image, draw_segmentation
from prediction import predict_image, load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_JSON_EXTENSIONS = {'json'}

def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

# Model
model = None
model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model():
    global model, model_loaded
    try:
        model_path = os.environ.get("MODEL_PATH", "model.pth")
        logging.info(f"Loading model from {model_path} on device: {device}")
        model = load_model(model_path, num_classes=10, device=device)
        model_loaded = True
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        model_loaded = False

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "Server is live"})

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device)
    }), 200 if model_loaded else 500

@app.route('/upload', methods=['POST'])
def upload():
    start_time = time()

    if not model_loaded:
        logging.error("Model not loaded, rejecting request")
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        if 'image' not in request.files or 'json' not in request.files:
            return jsonify({'error': 'Missing image or JSON file'}), 400

        image_file = request.files['image']
        json_file = request.files['json']

        if not image_file.filename or not json_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file type'}), 400

        if not allowed_file(json_file.filename, ALLOWED_JSON_EXTENSIONS):
            return jsonify({'error': 'Invalid JSON file type'}), 400

        unique_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{image_file.filename}")
        json_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{json_file.filename}")

        image_file.save(image_path)
        json_file.save(json_path)

        data, image = load_json_and_image(json_path, image_path)
        if data is None or image is None:
            return jsonify({"error": "Failed to load files"}), 400

        segmented_image = draw_segmentation(data, image)
        if segmented_image is None:
            return jsonify({"error": "Segmentation failed"}), 500

        segmented_image_filename = f"segmented_{unique_id}.jpg"
        segmented_image_path = os.path.join(RESULTS_FOLDER, segmented_image_filename)
        cv2.imwrite(segmented_image_path, segmented_image)

        # Prediction
        class_names = ["3VT", "ARSA", "AVSD", "Dilated Cardiac Sinus", "ECIF", "HLHS", "LVOT", "Normal Heart", "TGA", "VSD"]
        predicted_class = predict_image(segmented_image_path, model, class_names, device)

        base_url = request.url_root.rstrip('/')
        segmented_image_url = f"{base_url}/results/{segmented_image_filename}"

        duration = time() - start_time
        logging.info(f"Processed in {duration:.2f} seconds")

        return jsonify({
            "predictions": predicted_class,
            "annotations": data.get("shapes", []),
            "segmented_image": segmented_image_url,
            "processing_time_sec": round(duration, 2)
        })

    except Exception as e:
        logging.exception(f"Error in upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    initialize_model()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
