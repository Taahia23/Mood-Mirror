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