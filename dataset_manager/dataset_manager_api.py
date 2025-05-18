from flask import Flask, Response
import numpy as np
import dataset_manager_script

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of dataset manager'}, 200

@app.route('/get_images', methods=['GET'])
def get_images():
    images = dataset_manager_script.images
    def generate():
        yield images.tobytes()
    return Response(generate(), mimetype="application/octet-stream")

@app.route('/get_outputs', methods=['GET'])
def get_outputs():
    outputs = dataset_manager_script.outputs
    def generate():
        yield outputs.tobytes()
    return Response(generate(), mimetype="application/octet-stream")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)