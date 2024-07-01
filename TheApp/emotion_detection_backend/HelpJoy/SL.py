import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import librosa
import sounddevice as sd
import wavio
import os
import joblib

# Load pre-trained models and scaler
audio_model = load_model('../Models/best_model.h5')
video_model = load_model('../Models/emotion_detection_model.h5')
scaler = joblib.load('../Models/scaler.pkl')

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion classes
audio_emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
video_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set up session state variables for persistent data across reruns
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'confirmed_audio_emotion' not in st.session_state:
    st.session_state.confirmed_audio_emotion = None
if 'confirmed_image_emotion' not in st.session_state:
    st.session_state.confirmed_image_emotion = None

def record_audio(duration=5, fs=22050):
    """Record audio for a specific duration and save to a file."""
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    path = 'temp_audio.wav'
    wavio.write(path, recording, fs, sampwidth=2)
    st.session_state.audio_path = path

def preprocess_audio(audio_path):
    """Preprocess the audio file for prediction using loaded scaler and model."""
    audio, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1).reshape(1, -1)
    scaled_features = scaler.transform(mfcc_mean)
    return scaled_features.reshape(1, -1)

def capture_image():
    """Capture a single image from the default camera and detect faces using Haar Cascade."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        st.warning("No face detected. Try again.")
        return None
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        path = 'temp_image.jpg'
        cv2.imwrite(path, face_frame)
        break
    cap.release()
    st.session_state.image_path = path

def save_sample(sample_path, emotion, modality):
    """Save the sample into the respective folder for retraining, based on confirmed emotion and modality."""
    directory = f'{modality}/{emotion}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    base_filename = os.path.basename(sample_path)
    new_path = os.path.join(directory, base_filename)
    os.rename(sample_path, new_path)
    st.success(f"Saved {modality} sample as {new_path}")

st.title('Emotion Detection App')

# Audio recording and emotion detection
if st.button('Record Voice'):
    record_audio()
if st.session_state.audio_path:
    audio_data = preprocess_audio(st.session_state.audio_path)
    prediction = audio_model.predict(audio_data)
    detected_emotion = audio_emotions[np.argmax(prediction)]
    confirmed_emotion = st.selectbox('Confirm or Correct the Detected Emotion:', audio_emotions, index=audio_emotions.index(detected_emotion))
    if st.button('Save Audio Sample'):
        save_sample(st.session_state.audio_path, confirmed_emotion, 'audio')

# Image capturing and emotion detection
if st.button('Capture Image'):
    capture_image()
if st.session_state.image_path:
    img = cv2.imread(st.session_state.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    image_data = img.reshape(1, 64, 64, 1)
    prediction = video_model.predict(image_data)
    detected_emotion = video_emotions[np.argmax(prediction)]
    confirmed_emotion = st.selectbox('Confirm or Correct the Detected Emotion for Video:', video_emotions, index=video_emotions.index(detected_emotion))
    if st.button('Save Image Sample'):
        save_sample(st.session_state.image_path, confirmed_emotion, 'image')