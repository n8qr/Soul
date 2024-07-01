import numpy as np
import librosa

def process_audio(audio_path, model, scaler, emotion_labels):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfccs = scaler.transform([mfccs])
    prediction = model.predict(mfccs)
    emotion = np.argmax(prediction)
    return emotion_labels[emotion]
