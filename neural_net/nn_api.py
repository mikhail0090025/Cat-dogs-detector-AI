from flask import Flask, Response
import numpy as np
import nn_script

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of neural net'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)