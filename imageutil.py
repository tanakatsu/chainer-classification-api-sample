from PIL import Image
import base64
import cStringIO
import urllib2
import re


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
