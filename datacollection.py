import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "Data/very bad"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # --- Boundary check for cropping ---
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        # -----------------------------------

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hGap + hCal, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Show the main camera feed in full screen
    cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s") and hands and imgCrop.size != 0:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()