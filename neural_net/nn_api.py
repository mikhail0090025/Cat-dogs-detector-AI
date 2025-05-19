from flask import Flask, Response
import numpy as np
import nn_script

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of neural net'}, 200

@app.route('/one_epoch', methods=['GET'])
def one_epoch():
    return {'Response': 'This is a root of neural net'}, 200

@app.route('/inputs', methods=['GET'])
def inputs():
    return {'Response': nn_script.images}, 200

@app.route('/outputs', methods=['GET'])
def outputs():
    return {'Response': nn_script.outputs}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, ssl_context=None)