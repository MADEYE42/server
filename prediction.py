import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def get_advanced_model(num_classes):
    """
    Recreate the model architecture used in training.
    
    Args:
        num_classes (int): Number of classes in the model
    
    Returns:
        PyTorch model
    """
    # Use pretrained=False to avoid downloading ResNet50 weights
    model = models.resnet50(pretrained=False)

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
        model = get_advanced_model(num_classes)
        # Use weights_only=True for security and future compatibility
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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
        print(f"Error during prediction: {e}")
        return None

def main():
    MODEL_PATH = "model_path.pth"  # Change this to the actual model path
    DATA_DIR = "SplittedDataNew/train"  # Change this to your actual dataset directory
    IMAGE_PATH = "./segmented_output.jpg"  # Change this to the test image path

    device = torch.device("cpu")  # Force CPU for Render compatibility
    print(f"Using device: {device}")

    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')
    ])
    print(f"Detected classes: {class_names}")

    # Load the model
    model = load_model(MODEL_PATH, num_classes=len(class_names), device=device)

    if model is not None:
        predictions = predict_single_image(IMAGE_PATH, model, class_names, device)

        if predictions:
            print("\nTop 5 Predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['class']}: {pred['probability']:.2f}%")
        else:
            print("Prediction failed.")
    else:
        print("Model loading failed.")

if __name__ == "__main__":
    main()