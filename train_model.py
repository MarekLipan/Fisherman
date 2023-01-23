from ultralytics import YOLO
import os
import torch 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    model.train(data="custom.yaml", epochs=30)  # train the model