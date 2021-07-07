from flask import Flask, redirect, url_for, request, render_template, Response, jsonify
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import io

MODELS_PATH = './'
MNIST_MODEL = MODELS_PATH + "mnist.h5"
CNN_MODEL = MODELS_PATH + "srcnn.h5"

app = Flask(__name__)

@app.route('/')
def index():
    return "NN_API"

@app.route('/super-ress', methods=['POST'])
def gen():
    if request.files.get("image"):
        image = request.files['image']
        image = Image.open(image)
        try:
            generator = tf.keras.models.load_model(CNN_MODEL)
            arrayInput = np.array(image)
            input = tf.cast(arrayInput, tf.float32)[...,:3]
            input = (input/127.5) - 1
            image = tf.expand_dims(input, axis = 0)    
            genOutput = generator(image, training =  False) 
            generatedImageArray = genOutput[0, ...]

            generatedImage = Image.fromarray(np.uint8(((generatedImageArray+1)/2)*255), 'RGB')
            buffer = io.BytesIO()
            generatedImage.save(buffer,format="png")
            imageBuffer = buffer.getvalue()                     
        except:
            print("Error")

    return Response(imageBuffer, mimetype='image/png')

@app.route('/mnist', methods=['POST'])
def mnist():
    classNames = [0,1,2,3,4,5,6,7,8,9]
    if request.files.get("image"):
        try:
            image = request.files['image']
            image = Image.open(image)
            modelo = tf.keras.models.load_model(MNIST_MODEL)
            arrayInput = np.array(image)
            arrayInput = tf.expand_dims(arrayInput, axis = 0)
            print(arrayInput.shape)
            pred = modelo.predict(arrayInput) 
            number = np.array(classNames[np.argmax(pred)])
        except:
            print("Error")
            number = "error"
    
    return jsonify({'num': str(number)})


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)