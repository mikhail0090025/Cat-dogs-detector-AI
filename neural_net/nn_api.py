from flask import Flask, Response
import numpy as np
import nn_script

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of neural net'}, 200

@app.route('/one_epoch', methods=['GET'])
def one_epoch():
    try:
        if nn_script.main_model is None:
            return {'Response': f'Model is not ready yet. Wait...'}, 500
        nn_script.go_epochs(nn_script.main_model, 1)
        return {'Response': 'One epoch has successfully run!'}, 200
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route('/pass_epochs', methods=['POST'])
def pass_epochs(epochs_count):
    try:
        if nn_script.main_model is None:
            return {'Response': f'Model is not ready yet. Wait...'}, 500
        nn_script.go_epochs(nn_script.main_model, epochs_count)
        return {'Response': f'{epochs_count} epochs has successfully run!'}, 200
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route('/inputs', methods=['GET'])
def inputs():
    return {'Response': nn_script.images}, 200

@app.route('/outputs', methods=['GET'])
def outputs():
    return {'Response': nn_script.outputs}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, ssl_context=None)