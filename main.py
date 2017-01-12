# hello.py

from flask import Flask
import numpy as np
import chainer

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello Falsk'

if __name__ == '__main__':
    app.run()
