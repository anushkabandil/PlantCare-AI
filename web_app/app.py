import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model("../model/plant_disease_model.h5")

IMG_SIZE = (224,224)

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    return prediction.argmax()

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filepath = "temp.jpg"
        file.save(filepath)

        result = predict(filepath)

        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)