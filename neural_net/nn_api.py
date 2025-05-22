from flask import Flask, Response, request
import numpy as np
import nn_script
from nn_script import go_epochs, main
from flask_cors import CORS
import asyncio

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of neural net'}, 200

@app.route('/pass_epochs', methods=['POST'])
def pass_epochs():
    try:
        data = request.get_json()
        if not data:
            return {'Response': 'No JSON data provided'}, 400
        epochs_count = data.get('epochs_count')
        print(f"epochs_count: {epochs_count}")

        if epochs_count is None:
            return {'Response': f'Epochs count was none'}, 400
        if type(epochs_count) is not int:
            return {'Response': f'Epochs count was not an integer ({type(epochs_count).__name__}, {epochs_count})'}, 400
        if epochs_count <= 0:
            return {'Response': f'Epochs count cant be zero or less, it should be at least 1 ({epochs_count})'}, 400
        go_epochs(epochs_count)
        return {'Response': f'{epochs_count} epochs has successfully run!'}, 200
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route('/inputs', methods=['GET'])
def inputs():
    return {'Response': nn_script.images}, 200

@app.route('/outputs', methods=['GET'])
def outputs():
    return {'Response': nn_script.outputs}, 200

@app.route('/graphic', methods=['GET'])
def graphic():
    try:
        print(nn_script.all_losses)
        fig = nn_script.get_graphic(nn_script.all_losses, nn_script.all_val_losses, nn_script.all_accuracies, nn_script.all_val_accuracies)
        graph_json = fig.to_json()
        return {'graph': graph_json}
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

if __name__ == '__main__':
    asyncio.run(main())
    go_epochs(10)
    app.run(host='0.0.0.0', port=5001, ssl_context=None)