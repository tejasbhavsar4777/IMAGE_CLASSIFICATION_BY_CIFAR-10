from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

if not os.path.exists('./images'):
    os.makedirs('./images')


model = tf.keras.models.load_model('ICBT.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('./images', imagefile.filename)
    imagefile.save(image_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image)
    image = image.reshape((1, 32, 32, 3))
    image = image.astype('float32') / 255.0

    # Perform the prediction
    prediction = model.predict(image)

    # Convert the prediction to a readable label
    label = decode_prediction(prediction)

    # Render the result
    return render_template('index.html', prediction=label)

def decode_prediction(prediction):
    class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    predicted_class_index = prediction.argmax()
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    return f'{predicted_class} ({confidence:.2f}%)'

if __name__ == '__main__':
    app.run(port=3000, debug=True)
