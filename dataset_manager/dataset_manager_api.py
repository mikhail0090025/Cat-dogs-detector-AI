from flask import Flask, send_file, jsonify
import numpy as np
import dataset_manager_script
import os
import json
from PIL import Image, UnidentifiedImageError
import json
import requests
import io

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of dataset manager'}, 200

@app.route('/get_images', methods=['GET'])
def get_images():
    if not os.path.exists('numpy_dataset.npz'):
        return jsonify({'error': 'Dataset file not found'}), 404
    return send_file('numpy_dataset.npz', mimetype='application/zip', as_attachment=True, download_name='images.npz')

@app.route('/get_outputs', methods=['GET'])
def get_outputs():
    if not os.path.exists('numpy_dataset.npz'):
        return jsonify({'error': 'Dataset file not found'}), 404
    return send_file('numpy_dataset.npz', mimetype='application/zip', as_attachment=True, download_name='outputs.npz')

from flask import Flask, request
import aiohttp
from PIL import Image, UnidentifiedImageError
import io
import numpy as np

app = Flask(__name__)

@app.route("/image_to_inputs", methods=['GET'])
async def image_to_inputs():
    url = request.args.get('url')
    if not url:
        return {"error": "URL cannot be empty"}, 400

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                image_data = await response.read()

    except aiohttp.ClientError as e:
        return {"error": f"Failed to fetch image from URL: {e}"}, 400

    try:
        img = Image.open(io.BytesIO(image_data))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        try:
            img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
        except AttributeError:
            img_resized = img.resize((100, 100), Image.LANCZOS)  # Для старых версий PIL

        img_array = np.array(img_resized) / 255.0
        if img_array.shape != (100, 100, 3):
            return {"error": f"Expected image shape (100, 100, 3), got {img_array.shape}"}, 400
        
        if not (img_array.min() >= 0 and img_array.max() <= 1):
            return {"error": "Image array values out of expected range [0, 1]"}, 500

        img_array_list = img_array.tolist()
        
        return {"image": img_array_list}, 200
    
    except UnidentifiedImageError:
        return {"error": f"Error loading {url}: not a valid image"}, 400
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=None)