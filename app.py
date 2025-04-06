from flask import Flask, request, jsonify, send_from_directory
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from database import db, ScanResult
import scipy.special
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ✅ Set the model path from environment variable (default to 'models/deepfake_detector.tflite' if not set)
model_path = os.getenv("MODEL_PATH", "models/deepfake_detector.tflite")

# ✅ Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Log the model path (Make sure this is printing the correct model path)
logging.info(f"Loading model from path: {model_path}")

# ✅ Load TensorFlow Lite Model
model = None
try:
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # ✅ Dynamically Set Model Input Shape
    input_size = tuple(input_details[0]['shape'][1:3])  # (Height, Width)
    
    logging.info(f"✅ Model Loaded Successfully from: {model_path}")
    logging.info(f"📌 Model Input Shape: {input_size}")
except Exception as e:
    logging.error(f"🚨 ERROR: Model Loading Failed: {e}")


# ✅ Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake_results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ✅ Create Uploads Folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Home Route
@app.route('/')
def home():
    return jsonify({"message": "Deepfake Detection API is Running!"})

# ✅ API Status Route
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running!"})

# ✅ Store Deepfake Scan Results in Database
@app.route('/store_result', methods=['POST'])
def store_result():
    try:
        data = request.json
        filename = data.get('filename')
        prediction = data.get('prediction')
        confidence = data.get('confidence')

        if not filename or not prediction or confidence is None:
            return jsonify({"error": "Missing data"}), 400

        new_result = ScanResult(filename=filename, prediction=prediction, confidence=confidence)
        db.session.add(new_result)
        db.session.commit()

        return jsonify({"message": "Result stored successfully!"}), 201
    except Exception as e:
        logging.error(f"🚨 ERROR in storing results: {e}")
        return jsonify({"error": "Failed to store result"}), 500

# ✅ Upload Image Route
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Use secure_filename to avoid issues with filenames
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        return jsonify({"message": "Upload successful", "image_url": f"/images/{filename}"}), 201
    except Exception as e:
        logging.error(f"🚨 ERROR in uploading image: {e}")
        return jsonify({"error": "Image upload failed"}), 500

# ✅ Serve Uploaded Images
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Deepfake Detection Route
@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    if model is None:
        logging.error("Model is not loaded. Unable to proceed with detection.")
        return jsonify({"error": "Model failed to load"}), 500

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        allowed_extensions = {"png", "jpg", "jpeg"}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"Received file: {filename}")

        # Image preprocessing
        img = Image.open(file_path).convert("RGB").resize(input_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model.set_tensor(input_details[0]["index"], img_array)
        model.invoke()
        output_data = model.get_tensor(output_details[0]["index"])

        # ✅ Apply Sigmoid to Normalize Confidence
        raw_output = float(output_data[0][0])
        confidence = scipy.special.expit(raw_output)  # Ensures confidence is between 0 and 1
        confidence = round(confidence * 100, 2)  

        prediction = "Deepfake" if confidence > 50 else "Real"

        logging.info(f"Prediction: {prediction} with confidence: {confidence}%")

        return jsonify({
            "is_deepfake": prediction,
            "confidence": confidence,
            "processed_image": filename
        })

    except Exception as e:
        logging.error(f"🚨 ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true", host="0.0.0.0", port=port)
