import requests
from PIL import ImageGrab
import numpy as np
import cv2
import pyautogui
import pyaudio
import time
import random


# NOTE: it is necessary to enable stereo mix in the system audio settings and
# then set stereo mix as input device

GRAB_X = 850
GRAB_Y = 0
RATE = 44100
CHUNK = 1024
LISTEN_SEC = 60
ALPHA = 1
MAX_CATCHES = 100
FISHING_KEYBIND = "8"


def find_blob():
    """
    Takes screen, sents the image to blob detection app and moves
    the mouse cursor to the blob if detected
    """

    print("looking for blob")
    # take a screen
    img = ImageGrab.grab(bbox=(GRAB_X,GRAB_Y,1650,800))
    print("image taken")

    # Convert the image to numpy array
    img_np = np.array(img)
    # Encode the image in JPEG format
    _, img_bytes = cv2.imencode('.png', img_np)


    print("sending image")
    # Send the image to the Flask application for prediction
    response = requests.post("http://localhost:5000/predict", files={'image': ('image.png', img_bytes, 'image/png')}).json()
    print("image received")


    
    if len(response) == 0:
        print("Blob not found")
    else:
        x = int((response[0] + response[2]) / 2)
        y = int((response[1] + response[3]) // 2)

        click_point = (GRAB_X + x, GRAB_Y + y)

        # move to the bob
        pyautogui.moveTo(click_point, duration=0.3)
    print("cursor moved")

    return()


def listen_until_catch():
    """
    Uses the pyaudio library to listen until sound threshold is cross
    in the sound output (catch) happens
    """
    p = pyaudio.PyAudio()

    # open output stream
    output_stream = p.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=RATE,
                        input=True,
                        output_device_index=2)

    # record audio for RECORD_SECONDS
    start_time = time.time()
    while time.time() - start_time < LISTEN_SEC:
        data = output_stream.read(CHUNK)
        data = np.frombuffer(data, dtype=np.int16)
        loudness = np.abs(data).mean()
        # catch
        if loudness > ALPHA:
            return
        time.sleep(0.2)

    output_stream.stop_stream()
    output_stream.close()
    p.terminate()

def main():
    """
    Start cathing fish and continue until MAX_CATCHES
    """

    print("Start cathing")
    time.sleep(5)

    for i in range(MAX_CATCHES):

        # throw in
        pyautogui.keyDown(FISHING_KEYBIND)
        time.sleep(random.uniform(0.03, 0.07))
        pyautogui.keyUp(FISHING_KEYBIND)
        time.sleep(random.uniform(1.5, 2))

        find_blob()

        listen_until_catch()

        # pull out
        time.sleep(random.uniform(0.2, 0.5))
        pyautogui.click(button="right")
        time.sleep(random.uniform(1, 1.5))



    print(f"Catch {i}/{MAX_CATCHES}")

    print("Done")

    return

if __name__ == "__main__":
    main()






