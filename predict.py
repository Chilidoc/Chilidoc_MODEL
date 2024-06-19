from flask import Flask, request, jsonify
import numpy as np
import requests
from keras.models import load_model
from keras.preprocessing import image
import os
from PIL import Image

app = Flask(__name__)

model = load_model('best_model.h5')

# Function to preprocess the image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Function to predict the disease
def predict_disease(model, img_path, class_names):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    disease_name = class_names[predicted_class[0]]
    return disease_name

# Define class names as used during training
class_names = ['Healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowfish']

# Recommendations dictionary
recommendations = {
    'Healthy': {
        'Desc': 'The plant is healthy with no signs of disease.',
        'Treatment': 'No treatment needed.',
        'Prevention': 'Maintain good agricultural practices.'
    },
    'leaf curl': {
        'Desc': 'Leaf curl disease description here.',
        'Treatment': 'Apply appropriate insecticides.',
        'Prevention': 'Use resistant varieties and manage vector populations.'
    },
    'leaf spot': {
        'Desc': 'The plant is affected by a fungal infection causing spots on the leaves.',
        'Treatment': 'Apply fungicides as recommended.',
        'Prevention': 'Ensure proper spacing and air circulation.'
    },
    'whitefly': {
        'Desc': 'The plant is infested with whiteflies, which are small flying insects.',
        'Treatment': 'Use insecticidal soap or neem oil.',
        'Prevention': 'Introduce natural predators and maintain field hygiene.'
    },
    'yellowfish': {
        'Desc': 'The leaves of the plant are turning yellow due to a nutrient deficiency.',
        'Treatment': 'Apply iron chelates if caused by nutrient deficiency.',
        'Prevention': 'Monitor soil pH and nutrient levels regularly.'
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Save the image file locally
    img_path = 'uploaded_image.jpg'
    file.save(img_path)

    # Perform disease prediction
    disease_name = predict_disease(model, img_path, class_names)
    
    # Remove the locally saved image
    os.remove(img_path)

    # Retrieve recommendations based on prediction
    if disease_name in recommendations:
        treatment = recommendations[disease_name]['Treatment']
        prevention = recommendations[disease_name]['Prevention']
        desc = recommendations[disease_name]['Desc']
        response = {
            'Disease': disease_name,
            'Desc': desc,
            'Treatment': treatment,
            'Prevention': prevention
        }
    else:
        response = {'error': 'Disease not recognized. Please consult an expert.'}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
