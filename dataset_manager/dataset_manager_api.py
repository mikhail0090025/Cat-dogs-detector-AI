from flask import Flask, send_file, jsonify
import numpy as np
import dataset_manager_script
import os
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of dataset manager'}, 200

@app.route('/get_images', methods=['GET'])
def get_images():
    if not os.path.exists('numpy_dataset.npz'):
        return jsonify({'error': 'Dataset file not found'}), 404
    return send_file('numpy_dataset.npz', mimetype='application/octet-stream', as_attachment=True, download_name='images.npz')

@app.route('/get_outputs', methods=['GET'])
def get_outputs():
    if not os.path.exists('numpy_dataset.npz'):
        return jsonify({'error': 'Dataset file not found'}), 404
    return send_file('numpy_dataset.npz', mimetype='application/octet-stream', as_attachment=True, download_name='outputs.npz')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=None)