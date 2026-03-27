import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train_dir = "../dataset/fer2013/train"
test_dir = "../dataset/fer2013/test"

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)


test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)