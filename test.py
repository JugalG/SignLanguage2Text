import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

model_path = os.path.abspath("Model/keras_model.h5")
labels_path = os.path.abspath("Model/labels.txt")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
# classifier = Classifier(r"Model\keras_model.h5", r"Model\labels.txt")

classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300
# folder = "Data/C"
counter = 0
sentence = []
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M","N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # print("Hand Detected")
         # Check if imgCrop is not empty before resizing
        if not imgCrop.size == 0:
            aspectRatio = h / w


        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print("Prediction (Aspect Ratio > 1):", prediction, index)
            print(labels[index])
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print("Prediction (Aspect Ratio <= 1):", prediction, index)
            print(labels[index])

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # print("Showing Image")
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        sentence.append(labels[index])
        print(counter)
    elif key == ord("k"):
        counter +=1
        sentence.append(" ")
    elif key == 27:  # Press 'Esc' to exit
        print(sentence)
        break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the pre-trained model
# model = load_model("Model/keras_model.h5")

# # Load labels
# with open("Model/labels.txt", "r") as file:
#     labels = file.read().splitlines()

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()

#     # Preprocess the image for your model
#     img = cv2.resize(img, (224, 224))  # Adjust the size as needed
#     img = img / 255.0  # Normalize pixel values to be between 0 and 1

#     # Expand dimensions to match the model's expected input shape
#     img = np.expand_dims(img, axis=0)

#     # Make predictions
#     predictions = model.predict(img)
#     predicted_class_index = np.argmax(predictions)
#     prediction_label = labels[predicted_class_index]

#     # Display the results
#     cv2.putText(imgOutput, f"Prediction: {prediction_label}", (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)

#     if key == 27:  # Press 'Esc' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()
