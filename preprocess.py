
import json
import os
import cv2
import shutil
import random


def via2yolo(path: str):
    """
    Given a path to the via tool annotation .json, it creates a set of yolo acceptable annotation .txt files.
     It only works on rectagles and requires that the images are present in the same folder as annotations.
    """

    with open(os.path.join(path, "via_export_json.json")) as f:
        via_anns = json.load(f)

    for a in via_anns:
        filename = via_anns[a]["filename"]
        regions = via_anns[a]["regions"]
        assert len(regions) == 1
        object = regions[0]["shape_attributes"]
        assert object["name"] == "rect"

        img = cv2.imread(os.path.join(path, filename))
        Y, X, _ = img.shape

        x = (object['x'] + (object['width'] / 2)) / X
        y = (object['y'] + (object['height'] / 2)) / Y

        output_line = f"0 {x} {y} {object['width']/X} {object['height']/Y}"

        ann_filename = filename.split(".")[0] + ".txt"
        outpufile = open(os.path.join(path, ann_filename),"w")
        outpufile.writelines(output_line)
        outpufile.close()

    return

    
def train_val_split(path: str):
    """
    Splits all the images and their annotation files to train and val folder structure
    """

    with open(os.path.join(path, "all", "via_export_json.json")) as f:
        via_anns = json.load(f)

    for a in via_anns:
        filename = via_anns[a]["filename"]
        ann_filename = filename.split(".")[0] + ".txt"

        if random.random() <= 0.8:
            shutil.copyfile(os.path.join(path, "all", filename), os.path.join(path, "train/images", filename))
            shutil.copyfile(os.path.join(path, "all", ann_filename), os.path.join(path, "train/labels", ann_filename))
        else:
            shutil.copyfile(os.path.join(path, "all", filename), os.path.join(path, "val/images", filename))
            shutil.copyfile(os.path.join(path, "all", ann_filename), os.path.join(path, "val/labels", ann_filename))

    return


def clean_up_data(path: str):
    """
    Deletes the translated annotations and the train/val folders
    """

    filepath = os.path.join(path, "all")
    for f in os.listdir(filepath):
        if ".txt" in f:
            os.remove(os.path.join(filepath, f))

    filepath = os.path.join(path, "train/images")
    for f in os.listdir(filepath):
        os.remove(os.path.join(filepath, f))

    filepath = os.path.join(path, "train/labels")
    for f in os.listdir(filepath):
        os.remove(os.path.join(filepath, f))

    filepath = os.path.join(path, "val/images")
    for f in os.listdir(filepath):
        os.remove(os.path.join(filepath, f))

    filepath = os.path.join(path, "val/labels")
    for f in os.listdir(filepath):
        os.remove(os.path.join(filepath, f))

    filepath = os.path.join(path, "train/labels.cache")
    if os.path.exists(filepath):
        os.remove(filepath)

    filepath = os.path.join(path, "val/labels.cache")
    if os.path.exists(filepath):
        os.remove(filepath)

    return


#clean_up_data("datasets/data")
#via2yolo("datasets/data/all")
#train_val_split("datasets/data")




