import csv
import random
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import openai
from transformers import pipeline
import threading
import os
from scipy.io import wavfile
from pydub import AudioSegment
import pandas as pd

# Initialize Flask app
app = Flask(__name__, static_folder='../emotion-app/build', static_url_path='/')
CORS(app)

# Set up OpenAI API key
openai.api_key = ''  # Replace with your actual API key

# Load the entire model
model = load_model('Models/emotion_detection_model.h5')

# Load audio emotion detection model
audio_model = load_model('Models/Aud.h5')

# Initialize face detector
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the Haar Cascade file loaded correctly
if face_haar_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file.")

# Define emotions corresponding to the output labels for audio model
audio_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define emotions corresponding to the output labels for video model
video_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load BERT model for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize conversation state
conversation = []
current_emotion = None
video_analysis_enabled = False  # Flag to control video analysis

# CSV logging setup
log_file = 'res/emotion_log.csv'
menu_file = 'res/menu.csv'


# Function to log detected emotions
def log_emotion(emotion):
    with open(log_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([emotion])


# Function to preprocess the face region
def preprocess_frame(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    return faces_detected, gray_img


# Function to detect emotion continuously
def detect_emotion_continuously():
    global current_emotion, video_analysis_enabled
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    while True:
        if not video_analysis_enabled:
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        faces_detected, gray_img = preprocess_frame(frame)
        if len(faces_detected) > 0:
            for (x, y, w, h) in faces_detected:
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (64, 64))
                img_pixels = img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                current_emotion = video_emotions[max_index]
                log_emotion(current_emotion)
        cv2.waitKey(10)


# Start emotion detection in a separate thread
threading.Thread(target=detect_emotion_continuously, daemon=True).start()


# Function to get the GPT-3.5 response
def get_gpt3_response(prompt, conversation):
    conversation.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=300,
        temperature=1,
    )
    reply = response.choices[0].message['content']
    conversation.append({"role": "assistant", "content": reply})
    return reply


# Function to calculate random discount
def calculate_discount():
    menu = pd.read_csv(menu_file)
    random_item = menu.sample(n=1).iloc[0]
    item = random_item['Item']
    original_price = random_item['Price']
    discount_percentage = random.uniform(0, 20)  # Random discount up to 20%
    discounted_price = round(original_price * (1 - discount_percentage / 100), 2)
    return item, original_price, discounted_price, discount_percentage


# Function to preprocess audio
def preprocess_audio(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)

    # Ensure audio is mono (single channel)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio data
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    # Calculate target number of samples to match the target shape
    target_samples = 104 * 20  # 2080 samples to match the shape (104, 20)

    # Resample or pad the audio data to the target number of samples
    if len(audio_data) > target_samples:
        audio_data = audio_data[:target_samples]  # Truncate
    elif len(audio_data) < target_samples:
        padding = target_samples - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')  # Pad

    # Reshape audio data to fit the model's expected input shape (104, 20)
    audio_data = np.reshape(audio_data, (104, 20))

    # Expand dimensions to match model input (batch_size, height, width, channels)
    input_data = np.expand_dims(audio_data, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


# Function to test audio model
def test_audio_model(audio_path):
    input_data = preprocess_audio(audio_path)
    predictions = audio_model.predict(input_data)
    max_index = np.argmax(predictions[0])
    detected_emotion = audio_emotions[max_index]
    return detected_emotion


# Endpoint to handle audio emotion detection
@app.route('/api/audio_emotion', methods=['POST'])
def audio_emotion():
    global current_emotion
    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)

    try:
        # Convert audio to WAV format if it's not
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # Ensure mono, 16kHz, 16-bit

        # Limit the audio length to 10 seconds
        if len(audio) > 10000:
            audio = audio[:10000]

        audio.export(audio_path, format='wav')

        current_emotion = test_audio_model(audio_path)
        log_emotion(current_emotion)
    except Exception as e:
        print(f"Error in audio emotion detection: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)  # Clean up the temporary audio file

    return jsonify({"response": f"The detected emotion is {current_emotion}."})


# Endpoint to handle user visit or restart and provide initial GPT response
@app.route('/api/start', methods=['POST'])
def start_conversation():
    global conversation, current_emotion
    conversation = [
        {"role": "system", "content":
            "You are a friendly and helpful assistant. Interact in a nice and personal way. "
            "Communicate in English. I will tell you someone's emotions, "
            "and your goal is to talk to them in a way that suits their feelings."
         }
    ]
    if current_emotion:
        initial_prompt = f"The person is feeling {current_emotion}. How can you talk to them in a way that suits their feelings?"
        response = get_gpt3_response(initial_prompt, conversation)
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Unable to detect emotion at the moment. Please try again."})


# Endpoint to handle user input and get GPT response
@app.route('/api/message', methods=['POST'])
def handle_message():
    global conversation
    data = request.json
    user_input = data['message']
    analysis = sentiment_analyzer(user_input)
    detected_emotion = analysis[0]['label'].lower()
    log_emotion(detected_emotion)
    if analysis[0]['label'] == 'POSITIVE':
        item, original_price, discounted_price, discount_percentage = calculate_discount()
        response = f"You seem happy now! 😊 You get a {discount_percentage:.2f}% discount on {item}. Original price: ${original_price:.2f}, Discounted price: ${discounted_price:.2f}. Conversation ended."
        conversation = []  # Clear conversation history
    else:
        response = get_gpt3_response(user_input, conversation)
    return jsonify({"response": response})


# Function to generate frames for video streaming
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Endpoint to stream video
@app.route('/video_feed')
def video_feed():
    global video_analysis_enabled
    video_analysis_enabled = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Endpoint to stop video analysis
@app.route('/stop_video_analysis', methods=['POST'])
def stop_video_analysis():
    global video_analysis_enabled
    video_analysis_enabled = False
    return jsonify({"message": "Video analysis stopped"})


# Endpoint to handle text emotion detection
@app.route('/api/text_emotion', methods=['POST'])
def text_emotion():
    global conversation, current_emotion
    data = request.json
    user_input = data['text']
    analysis = sentiment_analyzer(user_input)
    detected_emotion = analysis[0]['label'].lower()
    current_emotion = detected_emotion
    log_emotion(current_emotion)

    if detected_emotion == 'positive':
        item, original_price, discounted_price, discount_percentage = calculate_discount()
        response = f"The person is feeling happy. 😊 They get a {discount_percentage:.2f}% discount on {item}. Original price: ${original_price:.2f}, Discounted price: ${discounted_price:.2f}."
    else:
        response = get_gpt3_response(
            f"The person is feeling {current_emotion}. How can you talk to them in a way that suits their feelings?",
            conversation)
    return jsonify({"response": response})


# Serve React frontend
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# Endpoint to provide emotion summary
@app.route('/api/emotion_summary', methods=['GET'])
def emotion_summary():
    try:
        if not os.path.exists(log_file):
            return jsonify({})

        df = pd.read_csv(log_file, header=None, names=['emotion'])
        emotion_counts = df['emotion'].value_counts().to_dict()
        return jsonify(emotion_counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)
