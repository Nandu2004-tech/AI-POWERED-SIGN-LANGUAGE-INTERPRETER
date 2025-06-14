import cv2
from cvzone.HandTrackingModule import HandDetector 
from cvzone.ClassificationModule import Classifier 
import numpy as np
import math 
import csv
from datetime import datetime 
import time
import os

# Setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) 
classifier = Classifier(
    "C:/Users/knand/Desktop/model/converted_keras/keras_model.h5",
    "C:/Users/knand/Desktop/model/converted_keras/labels.txt"
)
offset = 20
imgSize = 300
labels = [ "all the best","Hello","I love you","No", "Okay", "Please","Thank you","Very bad","Yes"]

# Create directory to save test crops 
if not os.path.exists("SavedCrops"):
    os.makedirs("SavedCrops")

# Open CSV file for logging
csv_file = open('C:/Users/knand/Desktop/sign language detection/sign_language_predictions.csv', mode='w', newline='') 
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Prediction", "Confidence"])

# FPS tracking 
pTime = 0
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
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 
        # Safe cropping
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        aspectRatio = h / w if w != 0 else 0

        try:
            if imgCrop.size == 0:
                raise Exception("Empty crop")
            if aspectRatio > 1: 
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize)) 
                wGap = math.ceil((imgSize - wCal) / 2) 
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w if w != 0 else 0
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal)) 
                hGap = math.ceil((imgSize - hCal) / 2) 
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False) 
            # Robust confidence extraction for 1D or 2D prediction
            if isinstance(prediction, (list, np.ndarray)) and len(prediction) > index:
                confidence = round(float(prediction[index]) * 100, 2)
            elif hasattr(prediction, "__len__") and len(prediction) > 0 and hasattr(prediction[0], "__len__"):
                confidence = round(float(prediction[0][index]) * 100, 2)
            else:
                confidence = 0.0
            # Draw results
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, f"{labels[index]} ({confidence}%)", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Log to CSV
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            csv_writer.writerow([timestamp, labels[index], confidence])
            csv_file.flush()

             # Show test images 
            cv2.imshow('ImageCrop', imgCrop) 
            cv2.imshow('ImageWhite', imgWhite)

            # Save crop automatically
            save_path = os.path.join(os.getcwd(), "SavedCrops", f"Crop_{time.time()}.jpg")
            success = cv2.imwrite(save_path, imgCrop)
            print("Image saved:", success, "at", save_path)

        except Exception as e:
            print("Error processing image:", e)

    # Show output and FPS 
    cTime = time.time()
    fps = 1 / (cTime - pTime + 0.001) 
    pTime = cTime
    cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Image', imgOutput) 

       

    # Keyboard controls
  
    key = cv2.waitKey(1)
    if key == ord('s') and hands:
        print("imgCrop shape:", imgCrop.shape)
        save_path = os.path.join(os.getcwd(), "C:/Users/knand/Desktop/sign language detection/SavedCrops", f"Crop_{time.time()}.jpg")
        success = cv2.imwrite(save_path, imgCrop)
        print("Image saved:", success, "at", save_path)
    elif key == ord('q'):
        break

# Cleanup
csv_file.close() 
cap.release() 
cv2.destroyAllWindows()