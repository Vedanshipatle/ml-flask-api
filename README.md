# 🧠 ML Model + Flask API (Text & Image Classification)

This project is Flask-based API that uses two open-source machine learning models:
- **Text Classification**: Classifies text sentiment (positive, negative, neutral) using a pre-trained BERT model.
- **Image Classification**: Classifies input images using a pre-trained ResNet18 model from PyTorch.

---

## 🚀 Features

- REST API using Flask
- Two endpoints:
  - `/predict-text`: accepts raw text in JSON
  - `/predict-image`: accepts image files (JPG/PNG)
- Returns structured JSON with prediction and confidence
- Works locally on any system with Python installed

---



# API Usage

📝 TEXT PREDICTION


URL: POST /predict-text
Input: JSON with a "text" field

Example request:
json
{
  "text": "I love learning new things!"
}
Example using Postman or curl:
bash

curl.exe -X POST http://127.0.0.1:5000/predict-text -H "Content-Type: application/json" -d "{\"text\":\"I am very happy\"}"
Example response:
json

{
  "success": true,
  "prediction": "positive",
  "confidence": 0.94
}

🖼️ IMAGE PREDICTION 


URL: POST /predict-image
Input: multipart/form-data with image field

Example using Postman:
Method: POST

URL: http://127.0.0.1:5000/predict-image

Body: form-data

Key: image

Type: File

Value: upload a .jpg or .png image

Example response:
json

{
  "success": true,
  "prediction": "Labrador retriever",
  "confidence": 0.91
}

# Model Info

Text Model:   🤗 HuggingFace: distilbert-base-uncased-finetuned-sst-2-english
Task:            Sentiment analysis
Image Model:     PyTorch: ResNet18 pretrained on ImageNet
Task:            Image classification 


# Requirements
Installed via requirements.txt:

Flask
torch
torchvision
transformers
pillow
requests

# Result JSON Format
Both endpoints return a response like:

{
  "success": true,
  "prediction": "label or list of labels",
  "confidence": 0.95
}

# Deployment:
Replit


