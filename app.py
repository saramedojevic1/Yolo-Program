from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploadovane_slike' 
 # Ensure this directory exists
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'kuglice.pt'  # Update this path to your model's weights

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image through YOLOv8 model
        results = model(filepath)
        for r in results:
            im_array = r.plot()  # Plotting in BGR
            im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB for PIL
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            im.save(output_path)  # Save the processed image

        return render_template('results.html', filename='uploadovane_slike/processed_' + filename)
    
  


if __name__ == '__main__':
    app.run(debug=True)
