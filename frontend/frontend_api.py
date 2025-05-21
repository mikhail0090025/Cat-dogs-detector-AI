from flask import Flask, render_template
import numpy as np
import frontend_script

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def root():
    return render_template('mainpage.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, ssl_context=None)