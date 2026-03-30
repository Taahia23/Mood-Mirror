import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/emotion_model.h5")