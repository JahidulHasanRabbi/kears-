from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model  # Add this import

import numpy as np
import os


Name = ['Bacterial Leaf Blight', 
        'Brown Spot',
        'Healthy', 
        'Leaf Blast', 
        'Leaf Scald', 
        'Narrow Brown Spot']

N=[]
for i in range(len(Name)):
    N+=[i]
normal_mapping=dict(zip(Name,N))
reverse_mapping=dict(zip(N,Name))
def mapper(value):
    return reverse_mapping[value]


app = Flask(__name__)

# Load the Keras model
model = load_model('model/DenseNet201.h5')


@app.route('/')
def hello():
    return render_template('predict.html')

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

    image_size = (224, 224)
    # Preprocess the image
    image = load_img(img_path, target_size=image_size)
    image=img_to_array(image)
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    prediction=model.predict(prediction_image)
    value=np.argmax(prediction)
    move_name=mapper(value)
    print(move_name)
    
    return jsonify({'class': move_name})
    os.remove(img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)