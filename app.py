import os
from flask import Flask, request, jsonify
from flask_ngrok3 import run_with_ngrok, get_host
from werkzeug.utils import secure_filename
from inference import process_image  # Import the inference logic

app = Flask(__name__)


# Define a folder to save uploaded images temporarily
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Save the uploaded image
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the image using the MMOCR inference function
    recognized_text = process_image(file_path)

    # Remove the image after processing if necessary
    os.remove(file_path)

    return jsonify({'recognized_text': recognized_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)