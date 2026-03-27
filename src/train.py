import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train_dir = "../dataset/fer2013/train"
test_dir = "../dataset/fer2013/test"