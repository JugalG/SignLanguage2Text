import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

model_path = os.path.abspath("Model/keras_model.h5")
labels_path = os.path.abspath("Model/labels.txt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

most_probable_alphabet = ""
max_confidence = 0.0

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image. Exiting.")
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = classifier.getPrediction(imgWhite, draw=False)[0]

            print("Prediction:", labels[index], "Confidence:", confidence)

            if confidence > max_confidence:
                most_probable_alphabet = labels[index]
                max_confidence = confidence

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, most_probable_alphabet, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        if key == 27:  # Press 'Esc' to exit
            break

    except Exception as e:
        print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()
