"""
webcam_detect.py
Real-time facial emotion detection using webcam feed.

Pipeline:
    1. Capture frame from webcam
    2. Detect face(s) using OpenCV Haar Cascade
    3. Crop, resize, normalize the face region
    4. Predict emotion using the trained CNN
    5. Draw bounding box + emotion label on the frame

Run from project root:
    python src/webcam_detect.py
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import EMOTIONS, IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.h5")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
    "Neutral": (200, 200, 200),
}


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first using src/train.py"
        )
    return load_model(MODEL_PATH)


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=-1)  # add channel dim
    face_img = np.expand_dims(face_img, axis=0)   # add batch dim
    return face_img


def main():
    model = load_emotion_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Starting webcam emotion detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            processed = preprocess_face(face_roi)

            predictions = model.predict(processed, verbose=0)[0]
            emotion_idx = int(np.argmax(predictions))
            emotion_label = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx]) * 100

            color = EMOTION_COLORS.get(emotion_label, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{emotion_label} ({confidence:.1f}%)"
            cv2.putText(
                frame, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.imshow("Emotion Detection - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
