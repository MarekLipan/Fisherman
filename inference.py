from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("runs/detect/train/weights/best.pt")
#model = YOLO("yolov8n.pt")

pred = model("c:/Users/marek/Desktop/Projects/datasets/data/train/images/0001.png")

print(pred)

#if __name__ == "__main__":
#    model.val(data="custom.yaml")