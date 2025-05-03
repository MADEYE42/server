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
    """
    try:
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
        logging.error(f"Error creating model: {str(e)}", exc_info=True)
        return None

def load_model(model_path, num_classes, device):
    """
    Load the trained model with the specified weights.
    """
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return None

        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logging.info(f"Model file size: {file_size_mb:.2f} MB")

        model = get_advanced_model(num_classes)
        if model is None:
            logging.error("Failed to initialize model architecture")
            return None

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("✅ Model loaded successfully.")
        return model

    except Exception as e:
        logging.error(f"❌ Error loading model: {str(e)}", exc_info=True)
        return None

def predict_image(image_path, model, device, class_names):
    """
    Make prediction on a single image.
    """
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
            logging.info(f"✅ Prediction: {prediction}")
            return prediction

    except Exception as e:
        logging.error(f"❌ Error predicting image: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    model_path = "path/to/model.pth"
    image_path = "path/to/test/image.jpg"
    class_names = ["Class1", "Class2", "Class3"]  # Update this as per your model
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, num_classes, device)
    if model:
        predict_image(image_path, model, device, class_names)
