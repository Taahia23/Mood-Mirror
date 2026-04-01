import streamlit as st
import cv2
import numpy as np
from src.predict import predict_emotion

st.title("Emotion Detection and Mental Health Support")

option = st.radio("Choose option:", ["Upload Image", "Webcam"])

# 📷 IMAGE MODE
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        emotion, suggestion = predict_emotion(img)

        st.image(img, channels="BGR")
        st.write(f"### Emotion: {emotion}")
        st.write(f"Suggestion: {suggestion}")

# 🎥 WEBCAM MODE
elif option == "Webcam":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        emotion, suggestion = predict_emotion(frame)

        cv2.putText(frame, emotion, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()