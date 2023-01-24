from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load the model
model = YOLO("runs/detect/train/weights/best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    image = request.files['image'].read()

    # Decode the image
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    # Make predictions using the model
    prediction = model(image)[0].boxes.data.tolist()

    # Needs to find exactly 1 blob
    if len(prediction) != 1:
        result = []
    else:
        result = prediction[0][:4]
        
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)