from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from torchvision import models, transforms
from PIL import Image
import io
import requests

# Create Flask app
app = Flask(__name__)



# Load image model
image_model = models.mobilenet_v2(pretrained=True)
image_model.eval()

# Load class labels from URL (ImageNet)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = requests.get(LABELS_URL).text.strip().split("\n")

# Image transform pipeline
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    input_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = image_model(input_tensor)
        _, predicted = outputs.max(1)
        label = classes[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    return jsonify({
        "success": True,
        "prediction": label,
        "confidence": round(confidence, 4)
    })







# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return jsonify({"message": "Flask API is working!"})

# TEXT PREDICTION ENDPOINT
@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"success": False, "error": "No text provided"}), 400

    result = sentiment_model(data['text'])[0]

    return jsonify({
        "success": True,
        "prediction": result['label'],
        "confidence": round(result['score'], 4)
    })

if __name__ == '__main__':
    app.run(debug=True)


