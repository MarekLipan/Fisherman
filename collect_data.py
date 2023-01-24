"""
Collecting data
"""

from PIL import ImageGrab
import os

DATA_PATH = "../datasets/data/all"

GRAB_X = 850
GRAB_Y = 0

# take a screen
img = ImageGrab.grab(bbox=(GRAB_X,GRAB_Y,1650,800))

# save iti
img_no = len(os.listdir(DATA_PATH))  # annotation json gives + 1
img.save(os.path.join(DATA_PATH,'{:04}.png'.format(img_no)))