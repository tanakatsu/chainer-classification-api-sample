from __future__ import print_function
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import models.VGG


def predict(img, n_candidates=3):
    class_labels = 100
    model = L.Classifier(models.VGG.VGG(class_labels))
    model.predictor.train = False
    serializers.load_npz('models/cifar100_model.npz', model)

    y = model.predictor(img.reshape(-1, 3, 32, 32))  # demension: 3 -> 4
    pred = F.softmax(y).data
    labels = pred[0].argsort()[-n_candidates:][::-1]
    scores = pred[0][labels]
    scores = map(lambda x: float(x), scores)  # Decimal -> float
    return zip(labels, scores)
