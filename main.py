# hello.py

from flask import Flask, request, render_template
import numpy as np
import chainer
from PIL import Image
import json
import imageutil

app = Flask(__name__)


@app.route('/')
def hello_world():
    img = Image.new('L', (64, 64))
    width, height = img.size
    print(img)
    print(width, height)
    return 'Hello.'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        url = request.args.get('url')
        data = request.args.get('data')
        if data:
            img = imageutil.read_image_from_base64(data)
        elif url:
            img = imageutil.read_image_from_url(url)
        else:
            return '''
            <h3>Error: Invalid parameter</h3>
            '''
        print(img.size)
        return 'ok'
    else:
        f = request.files['file']
        print(f.filename)
        img = imageutil.read_image_from_file(f)
        print(img.size)
        return 'ok'


@app.route('/predict.json', methods=['GET', 'POST'])
def predict_json():
    if request.method == 'GET':
        url = request.args.get('url')
        data = request.args.get('data')
        if data:
            img = imageutil.read_image_from_base64(data)
        elif url:
            img = imageutil.read_image_from_url(url)
        else:
            return json.dumps({'error': 'Invalid parameter'})
        print(img.size)
        return json.dumps({'size': img.size})
    else:
        f = request.files['file']
        print(f.filename)
        img = imageutil.read_image_from_file(f)
        print(img.size)
        return json.dumps({'size': img.size})


@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
