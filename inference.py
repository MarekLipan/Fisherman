import requests
import cv2

# Open an image file
with open("../datasets/data/all/0101.png", 'rb') as img:
    # Read the image file as binary data
    img_data = img.read()


# Send the image to the Flask application for prediction
response = requests.post("http://localhost:5000/predict", files={'image': ('image.png', img_data, 'image/png')}).json()

image = cv2.imread("../datasets/data/all/0101.png")

x = int((response[0] + response[2]) / 2)
y = int((response[1] + response[3]) // 2)

print(x)
print(y)

image = cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=-1)

cv2.imwrite("../datasets/data/all/0101_pred.png", image)