import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_advanced_model(num_classes):
    """
    Recreate the model architecture used in training.
    
    Args:
        num_classes (int): Number of classes in the model
    
    Returns:
        PyTorch model
    """
    try:
        # Use weights=None to avoid downloading ResNet18 weights and suppress warnings
        model = models.resnet18(weights=None)

        for param in model.parameters():
            param.requires_grad = True

        for param in list(model.layer3.parameters()) + list(model.layer4.parameters()):
            param.requires_grad = True

        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        return model
    except Exception as e:
        logging.error(f"Error creating model architecture: {str(e)}", exc_info=True)
        return None

def load_model(model_path, num_classes, device):
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes in the model
        device (torch.device): Device to load the model onto
    
    Returns:
        Loaded PyTorch model or None if there's an error
    """
    try:
        # Check if the model file exists and log its size
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return None
            
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logging.info(f"Model file found at {model_path}, size: {file_size_mb:.2f} MB")
        
        # Create the model architecture
        model = get_advanced_model(num_classes)
        if model is None:
            logging.error("Failed to create model architecture")
            return None
            
        # Load with error handling and explicit options
        try:
            # Use weights_only=True for security and future compatibility
            state_dict = torch.load(
                model_path, 
                map_location=device, 
                weights_only=True
            )
            model.load_state_dict(state_dict)
            logging.info("State dict loaded successfully")
        except RuntimeError as e:
            logging.error(f"Runtime error loading state dict: {str(e)}")
            # Try alternative loading method for older saved models
            logging.info("Trying alternative loading method...")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            logging.info("State dict loaded with alternative method")
            
        model = model.to(device)
        model.eval()
        logging.info(f"Model successfully loaded and moved to {device}")
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        return None

def predict_single_image(image_path, model, class_names, device):
    """
    Predict the class of a single image.
    
    Args:
        image_path (str): Path to the image file
        model (torch.nn.Module): Loaded PyTorch model
        class_names (list): List of class names
        device (torch.device): Device to run prediction on
    
    Returns:
        Prediction results or None if there's an error
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            logging.error(f"Image file not found at {image_path}")
            return None
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_k_probs, top_k_indices = torch.topk(probabilities, k=5)

            top_k_probs = top_k_probs.cpu().numpy()[0]
            top_k_indices = top_k_indices.cpu().numpy()[0]

            results = [{
                'class': class_names[idx],
                'probability': float(prob * 100)
            } for prob, idx in zip(top_k_probs, top_k_indices)]

            return results
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        return None

def main():
    MODEL_PATH = os.environ.get("MODEL_PATH", "model_path.pth")
    DATA_DIR = "SplittedDataNew/train"  # Change this to your actual dataset directory
    IMAGE_PATH = "./segmented_output.jpg"  # Change this to the test image path

    device = torch.device("cpu")  # Force CPU for Render compatibility
    logging.info(f"Using device: {device}")

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        return

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found at {DATA_DIR}")
        class_names = ["3VT", "ARSA", "AVSD", "Dilated Cardiac Sinus", "ECIF", "HLHS", "LVOT", "Normal Heart", "TGA", "VSD"]
    else:
        class_names = sorted([
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')
        ])
    
    logging.info(f"Using classes: {class_names}")

    # Load the model
    model = load_model(MODEL_PATH, num_classes=len(class_names), device=device)

    if model is None:
        logging.error("Model loading failed.")
    else:
        # Check if test image exists
        if not os.path.exists(IMAGE_PATH):
            logging.error(f"Test image not found at {IMAGE_PATH}")
            return
            
        predictions = predict_single_image(IMAGE_PATH, model, class_names, device)

        if predictions:
            logging.info("Top 5 Predictions:")
            for i, pred in enumerate(predictions, 1):
                logging.info(f"{i}. {pred['class']}: {pred['probability']:.2f}%")
        else:
            logging.error("Prediction failed.")

if __name__ == "__main__":
    main()