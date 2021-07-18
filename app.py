import base64
from io import BytesIO

import PIL
import torch
from PIL import Image
from flask import Flask, render_template, request
from flask_cors import CORS
from numpy import genfromtxt

import test
from src.utils.head_view import head_view
from src.utils.helpers import digitIterator

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
    return "OK"


import numpy as np


def sample_attentions():
    import numpy
    attentions = numpy.genfromtxt("data/atts.csv", delimiter=",").reshape((-1, 128))
    pred = '23102340234623753223602381234123662346235123752340238123462346238123352366234223812357233523672325236623063223402340238123523223442367230723252381235923672346237523402381322404240432241024093224042404'
    di = digitIterator(pred)
    tot_atts = []
    pred_str = di.get_str()
    for i in range(len(pred_str)):
        s, e = di.get_ith_idxs(i)
        att = np.mean(attentions[s + 2:e + 3], axis=0)
        tot_atts.append(torch.from_numpy(att))

    attentions = torch.stack(tot_atts)
    cross_attention = attentions.unsqueeze(0).unsqueeze(0)
    decoder_tokens = [c for c in pred_str]
    return cross_attention, decoder_tokens


@app.route('/ocr', methods=["GET", "POST"])
def ocr():
    if request.method == "POST":

        IMAGE_URL = "./current.jpg"
        if 'file' not in request.files and request.files['file'].filename != '':
            return "NO IMG FOUND"

        file = request.files['file']
        file.save(IMAGE_URL)

        image = Image.open(IMAGE_URL)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        pred_text, _, _ = test.test(pil_image=image)
        pred_text = pred_text.replace("-", " ")


    else:
        pred_text = None
        base64_img = None

    return render_template("ocr-view.html", pred_text=pred_text, base64_img=base64_img)


def getData(BASE_URL):
    # attentions
    attentions = genfromtxt(BASE_URL + '/atts.csv', delimiter=',').reshape((-1, 128))
    # image data:
    image = Image.open(BASE_URL + "/image.PNG")
    rot_image = image.rotate(-90, PIL.Image.NEAREST, expand=1)
    buffered = BytesIO()
    rot_image.save(buffered, format="PNG")
    image_64_encode = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # pred
    with open(BASE_URL + '/word.txt', 'r') as f:
        lines = f.readlines()

    lines = [x.strip() for x in lines]
    pred, ground = lines
    pred_di = digitIterator(pred)
    pred_str = pred_di.get_str()

    ground_di = digitIterator(ground)
    ground_str = ground_di.get_str()

    tot_atts = []
    for i in range(len(pred_str)):
        s, e = pred_di.get_ith_idxs(i)
        att = np.mean(attentions[s + 2:e + 3], axis=0)
        tot_atts.append(torch.from_numpy(att))

    attentions = torch.stack(tot_atts)
    cross_attention = attentions.unsqueeze(0).unsqueeze(0)

    return image_64_encode, cross_attention, pred_str, ground_str


@app.route('/attn/<idx>', methods=["GET", "POST"])
def show_attention(idx=99):
    BASE_URL = f"data/predict-1-{idx}"

    image_64_encode, cross_attention, pred_str, _ = getData(BASE_URL)

    decoder_tokens = [c for c in pred_str]
    encoder_tokens = [''] * cross_attention.shape[-1]

    cross_attention = (cross_attention,)
    params = head_view(cross_attention=cross_attention, encoder_tokens=encoder_tokens,
                       decoder_tokens=decoder_tokens)

    return render_template("head-view.html", params=params, base64_img=image_64_encode)


if __name__ == '__main__':
    app.run(debug=True)
    # sample_attentions()
