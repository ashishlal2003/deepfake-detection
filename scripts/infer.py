import torch
import torch.nn as nn
import cv2
import numpy as np
from model import DeepfakeDetectorCNN

def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32)

    return img

def load_model(model_path, device):
    model = DeepfakeDetectorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device=device)
    model.eval()
    return model

def predict(model, image_path, device):
    img = preprocess_image(image_path)
    img = img.to(device=device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    class_labels = ['Real', 'Fake']
    return class_labels[predicted.item()]

if __name__ == '__main__':
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference Script")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = load_model(args.model, device)

    # Perform inference
    try:
        prediction = predict(model, args.image, device)
        print(f"Prediction: The image is {prediction}.")
    except Exception as e:
        print(f"Error: {e}")