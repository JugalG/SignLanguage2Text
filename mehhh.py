import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load the pre-trained hand tracking model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the pre-trained sign language recognition model
model = load_model('Model\keras_model.h5')  # Replace with the actual path to your model

# Define the mapping of classes to sign language gestures
class_mapping = {
    0: 'A',
    1: 'B',
    # Add more mappings as needed
}

# Function to preprocess the hand image for prediction
def preprocess_image(image):
    # Preprocess your image as needed (resize, normalize, etc.)
    # ...

    # Return the preprocessed image
    return preprocessed_image

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands in the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the region of interest (ROI) around the hand
            hand_bbox = cv2.boundingRect(np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]))
            x, y, w, h = hand_bbox
            hand_roi = frame[y:y+h, x:x+w]

            # Preprocess the hand image for prediction
            processed_hand = preprocess_image(hand_roi)

            # Make predictions using the pre-trained sign language recognition model
            predictions = model.predict(np.expand_dims(processed_hand, axis=0))
            predicted_class = np.argmax(predictions)

            # Map the predicted class to the corresponding sign language gesture
            sign_language_gesture = class_mapping.get(predicted_class, 'Unknown')

            # Display the results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {sign_language_gesture}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
