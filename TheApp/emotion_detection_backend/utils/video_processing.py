import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

# Load the pre-trained model
model_fer = load_model('Models/emotion_detection_model.h5')
face_haar_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_frame(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    return faces_detected, gray_img

def detect_emotion(frame):
    faces_detected, gray_img = preprocess_frame(frame)
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        img_pixels = roi_gray.astype("float32") / 255.0
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)

        predictions = model_fer.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        return emotions[max_index]
    return None

def get_most_frequent_emotion(emotion_list):
    if emotion_list:
        return Counter(emotion_list).most_common(1)[0][0]
    return None
