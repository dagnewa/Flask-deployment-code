from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

import pathlib

# Keras
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\dagne\\Desktop\\Image-Classification-on-Flask-master\\image classification\\my_model.h5'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Download model if not present
while not pathlib.Path(MODEL_PATH).is_file():
    print(f'Model {MODEL_PATH} not found. Downloading...')
    #wget.download(MODEL_URL)

# Define a flask app
app = Flask(__name__)

# Define upload path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The skin is a diseased photosensitization.Removing the animal from sunlight "
    elif preds == 1:
        preds = "The skin is a diseased photosensitization.Removing the animal from sunlight"
    elif preds == 2:
        preds = "The skin is a diseased photosensitization.Removing the animal from sunlight"
    else:
        preds = "The skin is a diseased photosensitization.Removing the animal from sunlight:"


    return preds


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    if request.method == 'POST':
        # Get the file from post request
        print(request.files, request.form, request.args)
        f = None
        if 'image' in request.files: f = request.files['image']
        if f:
            # Save the file to ./uploads
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)
            result = preds
            return render_template('index.html', result=result, img=secure_filename(f.filename))
        return render_template('index.html', result=None, err='Failed to receive file')
    # First time
    return render_template('index.html', result=None)


if __name__ == '__main__':
    app.run(port=5001, debug=True)