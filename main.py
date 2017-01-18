from flask import Flask, request, render_template
import json
import pickle
import imageutil
# import mnist
import cifar

app = Flask(__name__)

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


def load_labels():
    with open('models/cifar100_labels.pkl') as f:
        data = pickle.load(f)
    labels = {}
    for i, label in enumerate(data['fine_label_names']):
        labels[i] = label
    return labels


def classify(img, n_candidates=3):
    resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    input_img = imageutil.to_np_data_array(resized)
    labels = load_labels()
    if n_candidates < 1:
        n_candidates = 1
    elif n_candidates >= len(labels):
        n_candidates = len(labels)
    results = cifar.predict(input_img, n_candidates)
    label_scores = [(labels[result[0]], result[1]) for result in results]
    return label_scores


@app.route('/')
def hello_world():
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
        if request.args.get('top'):
            n_candidates = int(request.args.get('top'))
        else:
            n_candidates = 3

        label_scores = classify(img, n_candidates)
        return render_template('result.html', results=label_scores, image=imageutil.encode_base64(img))
    else:
        f = request.files['file']
        if request.form and request.form['top']:
            n_candidates = int(request.form['top'])
        else:
            n_candidates = 3

        img = imageutil.read_image_from_file(f)
        label_scores = classify(img, n_candidates)
        return render_template('result.html', results=label_scores, image=imageutil.encode_base64(img))


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

        if request.args.get('top'):
            n_candidates = int(request.args.get('top'))
        else:
            n_candidates = 3
        label_scores = classify(img, n_candidates)
        return json.dumps(label_scores)
    else:
        f = request.files['file']

        if request.form and request.form['top']:
            n_candidates = int(request.form['top'])
        else:
            n_candidates = 3
        img = imageutil.read_image_from_file(f)
        label_scores = classify(img, n_candidates)
        return json.dumps(label_scores)


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload2')
def upload_by_drag_and_drop():
    return render_template('upload_drop.html')


@app.route('/labels.json')
def show_labels():
    labels = load_labels()
    return json.dumps(labels)

if __name__ == '__main__':
    app.debug = True
    app.run()
