import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "../dataset/test",
    target_size=IMG_SIZE,
    class_mode="categorical"
)

model = tf.keras.models.load_model("plant_disease_model.h5")

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)