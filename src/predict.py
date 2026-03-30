import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/emotion_model.h5")

labels = ["Angry", "Happy", "Neutral", "Sad"]

suggestions = {
    "Angry": "Try deep breathing",
    "Happy": "Keep smiling",
    "Neutral": "Stay positive",
    "Sad": "Take a break & talk to someone"
}

def predict_emotion(img):
    img = cv2.resize(img, (48,48))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    gray = np.reshape(gray, (1,48,48,1))

    pred = model.predict(gray)
    emotion = labels[np.argmax(pred)]

    return emotion, suggestions[emotion]