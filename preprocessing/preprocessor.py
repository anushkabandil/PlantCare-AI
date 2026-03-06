import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../dataset/New Plant Diseases Dataset(Augmented)/train"
valid_dir = "../dataset/New Plant Diseases Dataset(Augmented)/valid"
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data should not be augmented
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

# Load validation images
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

print("Training samples:", train_generator.samples)
print("Validation samples:", valid_generator.samples)