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
