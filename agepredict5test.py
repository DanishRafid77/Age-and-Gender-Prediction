import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\danis\Downloads\age_gender_model.h5")

# Function to preprocess face image
def preprocess_face(face):
    face = cv2.resize(face, (128, 128))  # Resize to match model input size
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]

        # Preprocess face
        preprocessed_face = preprocess_face(face)

        # Predict age and gender
        age_prediction, gender_prediction = model.predict(preprocessed_face)
        predicted_age = int(age_prediction[0][0])
        predicted_gender = "Male" if np.argmax(gender_prediction[0]) == 0 else "Female"

        # Draw rectangle around facex
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Overlay predictions on the frame
        label = f"Age: {predicted_age}, Gender: {predicted_gender}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the video feed
    cv2.imshow("Age and Gender Prediction", frame)

    # Break the loop on 'x' key press
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
