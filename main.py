from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import json
import imageutil
import mnist

app = Flask(__name__)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


@app.route('/')
def hello_world():
    img = Image.new('L', (64, 64))
    width, height = img.size
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
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        label, score = mnist.predict(np.asarray(resized).astype(np.float32) / 255)
        return render_template('result.html', label=label, score=score, image=imageutil.encode_base64(img))
    else:
        f = request.files['file']
        img = imageutil.read_image_from_file(f)
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        label, score = mnist.predict(np.asarray(resized).astype(np.float32) / 255)
        return render_template('result.html', label=label, score=score, image=imageutil.encode_base64(img))


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
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        label, score = mnist.predict(np.asarray(resized).astype(np.float32) / 255)
        return json.dumps({'label': label, 'score': score})
    else:
        f = request.files['file']
        print(f.filename)
        img = imageutil.read_image_from_file(f)
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        label, score = mnist.predict(np.asarray(resized).astype(np.float32) / 255)
        return json.dumps({'label': label, 'score': score})


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload2')
def upload_by_drag_and_drop():
    return render_template('upload_drop.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
