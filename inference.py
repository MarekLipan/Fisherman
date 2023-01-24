import requests
from PIL import ImageGrab
import numpy as np
import cv2
import pyautogui

GRAB_X = 850
GRAB_Y = 0

# take a screen
img = ImageGrab.grab(bbox=(GRAB_X,GRAB_Y,1650,800))

# Convert the image to numpy array
img_np = np.array(img)
# Encode the image in JPEG format
_, img_bytes = cv2.imencode('.png', img_np)

# Send the image to the Flask application for prediction
response = requests.post("http://localhost:5000/predict", files={'image': ('image.png', img_bytes, 'image/png')}).json()

if len(response) == 0:
    print("Blob not found")
else:
    x = int((response[0] + response[2]) / 2)
    y = int((response[1] + response[3]) // 2)

    click_point = (GRAB_X + x, GRAB_Y + y)

    # move to the bob
    pyautogui.moveTo(click_point, duration=0.3)