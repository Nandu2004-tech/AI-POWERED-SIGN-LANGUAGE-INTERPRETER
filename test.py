import cv2
from cvzone.HandTrackingModule import HandDetector 
from cvzone.ClassificationModule import Classifier 
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/knand/Desktop/model/converted_keras/keras_model.h5",
    "C:/Users/knand/Desktop/model/converted_keras/labels.txt"
)
offset = 20
imgSize = 300
labels = [ "all the best","Hello","I love you","No", "Okay", "Please","Thank you","Very bad","Yes"]
cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # Safe boundary check for cropping
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w if w != 0 else 0
        if aspectRatio > 1: 
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize)) 
            wGap = math.ceil((imgSize-wCal)/2) 
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w if w != 0 else 0
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal)) 
            hGap = math.ceil((imgSize - hCal) / 2) 
            imgWhite[hGap: hGap + hCal, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0,255,0), 4)
        cv2.imshow('ImageCrop', imgCrop) 
        cv2.imshow('ImageWhite', imgWhite)
    cv2.imshow('Image', imgOutput) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()