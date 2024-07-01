import cv2
import tensorflow as tf
import numpy as np

# Load your trained emotion detection model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define the list of emotions
video_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the Haar Cascade for face detection
# Correct initialization of the cascade classifier
# Specify the exact path to the cascade file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the captured frame into gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over the face detections
    for (x, y, w, h) in faces:
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels
        roi_gray = gray[y:y + h, x:x + w]
        resized_frame = cv2.resize(roi_gray, (64, 64))
        reshaped_frame = resized_frame.reshape(1, 64, 64, 1)

        # Normalize the pixel values
        reshaped_frame = reshaped_frame / 255.0

        # Perform emotion prediction
        prediction = model.predict(reshaped_frame)
        max_index = int(np.argmax(prediction))
        predicted_emotion = video_emotions[max_index]

        # Draw rectangle around the face and write the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
