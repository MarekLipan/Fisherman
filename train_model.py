from ultralytics import YOLO
import os
#import torch 
#print(torch.cuda.is_available())

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    model.train(data="custom.yaml", epochs=10)  # train the model