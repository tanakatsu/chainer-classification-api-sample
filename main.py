# hello.py

from flask import Flask
import numpy as np
import chainer
from PIL import Image

app = Flask(__name__)


@app.route('/')
def hello_world():
    img = Image.new('L', (64, 64))
    width, height = img.size
    print(img)
    print(width, height)
    return 'hello Falsk'

if __name__ == '__main__':
    app.debug = True
    app.run()
