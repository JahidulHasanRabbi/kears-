from flask import Flask, request, jsonify, json, Response
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np
import os


Name = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

N=[]
for i in range(len(Name)):
    N+=[i]
normal_mapping=dict(zip(Name,N))
reverse_mapping=dict(zip(N,Name))
def mapper(value):
    return reverse_mapping[value]


app = Flask(__name__)

# Load the Keras model
model = keras.models.load_model('model\DenseNet201.h5')


@app.route('/')
def hello():
    resphone = Response("Hello World!", status=200, mimetype='json')
    return resphone

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was sent with the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    uploaded_file = request.files['file']

    # Check if the file has a name
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image temporarily
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    img_path = os.path.join(upload_folder, uploaded_file.filename)
    uploaded_file.save(img_path)

    # Preprocess the image
    image = load_img(img_path)
    image=img_to_array(image)
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    prediction=model.predict(prediction_image)
    value=np.argmax(prediction)
    move_name=mapper(value)
    
    respons = Response({"Predication": move_name}, status=200, mimetype='json')

    return respons

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)