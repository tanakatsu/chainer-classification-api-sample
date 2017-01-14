#!/usr/bin/env python
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def predict(img, top=3):
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz('model/mnist_model.npz', model)

    # Predict
    y = model.predictor(img.reshape(-1, 784))
    pred = F.softmax(y).data
    labels = pred[0].argsort()[-top:][::-1]
    scores = pred[0][labels]
    scores = map(lambda x: float(x), scores)  # Decimal -> float
    return zip(labels, scores)
