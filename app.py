import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained CNN model
model = load_model('model/cnn_model.h5')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in request"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocess the uploaded image
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict with CNN model
            score = model.predict(img_array)[0][0]
            print(f" Model raw score: {score}")

            # Determine prediction class
            predicted_class = 'Malignant' if score >= 0.5 else 'Benign'

            # Map score to stage and survival rate (educational mapping)
            if score < 0.3:
                stage = 'No Cancer Detected'
                survival_rate = '99%'
            elif score < 0.5:
                stage = 'Stage I'
                survival_rate = '90%'
            elif score < 0.7:
                stage = 'Stage II'
                survival_rate = '80%'
            elif score < 0.85:
                stage = 'Stage III'
                survival_rate = '60%'
            else:
                stage = 'Stage IV'
                survival_rate = '40%'

            return render_template(
                'result.html',
                prediction=predicted_class,
                stage=stage,
                survival_rate=survival_rate,
                filename=filename
            )

        except Exception as e:
            return f"Error processing image: {str(e)}"

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
