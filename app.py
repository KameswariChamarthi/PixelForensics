from flask import Flask, request, jsonify, send_from_directory
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from database import db, ScanResult
import scipy.special 
from flask_cors import CORS

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/detect-deepfake": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake_results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create Uploads Folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load TensorFlow Lite Model
try:
    model_path = r"D:\New folder\deepfake_detection\deepfake_detector.tflite"
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    input_size = tuple(input_details[0]['shape'][1:3])  # (Height, Width)
    logging.info(f"✅ Model Loaded Successfully. Input Size: {input_size}")

except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None

# Home Route
@app.route('/')
def home():
    return jsonify({"message": "API is running!"})

# Deepfake Detection Route
@app.route('/detect-deepfake', methods=['POST'])
def detect_deepfake():
    if model is None:
        return jsonify({"error": "Model loading failed"}), 500

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        allowed_extensions = {"png", "jpg", "jpeg"}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({"error": "Invalid file type"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB").resize(input_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model.set_tensor(input_details[0]["index"], img_array)
        model.invoke()
        output_data = model.get_tensor(output_details[0]["index"])

        # Applying sigmoid for confidence
        raw_output = float(output_data[0][0])
        confidence = scipy.special.expit(raw_output)
        confidence = round(confidence * 100, 2)

        prediction = "Deepfake" if confidence > 50 else "Real"

        return jsonify({
            "is_deepfake": prediction,
            "confidence": confidence,
            "processed_image": file.filename
        })

    except Exception as e:
        logging.error(f"Error in deepfake detection: {e}")
        return jsonify({"error": "Detection failed"}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
