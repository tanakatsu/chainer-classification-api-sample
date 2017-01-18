from PIL import Image
import base64
import cStringIO
import urllib2
import re
import numpy as np


def read_image_from_file(file):
    return Image.open(cStringIO.StringIO(file.read()))


def read_image_from_data(data):
    return Image.open(cStringIO.StringIO(data))


def read_image_from_url(url):
    img = urllib2.urlopen(url).read()
    return Image.open(cStringIO.StringIO(img))


def read_image_from_base64(data):
    data = re.sub(r'^data:.+;base64,', '', data)
    decoded = base64.b64decode(data)
    return Image.open(cStringIO.StringIO(decoded))


def encode_base64(img):
    buf = cStringIO.StringIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue())
    return 'data:image/jpeg;base64,' + encoded


def to_pil_image(np_img):
    # np_img dimension: (height, width, channel)
    return Image.fromarray((np_img * 255).astype(np.uint8))
