import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import requests
import json

model_path = os.path.abspath("Model/keras_model.h5")
labels_path = os.path.abspath("Model/labels.txt")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300
counter = 0 
sentence = []

with open("Model/labels.txt", "r") as file:
    labels_lines = file.read().splitlines()
labels = [line.split(' ', 1)[1] for line in labels_lines]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Debug prints
        # print(f"imgCrop size: {imgCrop.shape if imgCrop is not None else None}")
        
        if imgCrop is not None and not imgCrop.size == 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                
                # Debug prints
                # print(f"hCal: {hCal}")
                
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            prediction, index = classifier.getPrediction(imgResize, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgResize)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        sentence.append(labels[index])
        print(labels[index])
    elif key == ord("k"):
        counter += 1
        sentence.append(" ")
        print(labels[index])
    elif key == 27:  # Press 'Esc' to exit
        combined_string = ''.join(sentence)
        print("sentence typed:", combined_string)

        url = "https://jobwave-careerfair.glitch.me/getSentence"
        data = {"combined_string": combined_string}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            print("Response data:", response.json()) 
        else:
            print(f"POST request failed with status code: {response.status_code}")
        break

cap.release()
cv2.destroyAllWindows()
