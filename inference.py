from ultralytics import YOLO
import os
from PIL import Image
from numpy import asarray

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("runs/detect/train/weights/best.pt")

#results = model()
