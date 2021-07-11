import base64
import tempfile
from io import BytesIO

import sys;

# from run import tester

# sys.path.append("..\..\seq2seq-attention-ocr-pytorch")
import PIL
from PIL import Image

from flask import Flask, render_template, request, redirect

import test

from flask_cors import CORS, cross_origin

from src.utils.head_view import head_view
import torch

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
    return "OK"


def sample_attentions():
    cross_attention = (torch.rand(1, 1, 5, 3),)
    encoder_tokens = ['1'] * 3
    decoder_tokens =  [c for c in 'सुमन्'] #['A'] * 5

    return cross_attention, encoder_tokens, decoder_tokens


@app.route('/attn', methods=["GET", "POST"])
def show_attention():

    if request.method == "POST":
        IMAGE_URL = "./current.jpg"
        if 'file' not in request.files and request.files['file'].filename != '':
            return "NO IMG FOUND"

        file = request.files['file']
        file.save(IMAGE_URL)

        image = Image.open(IMAGE_URL)
        rot_image = image.rotate(-90, PIL.Image.NEAREST, expand=1)
        buffered = BytesIO()
        rot_image.save(buffered, format="JPEG")
        image_64_encode = base64.b64encode(buffered.getvalue()).decode("utf-8")

        cross_attention, encoder_tokens, decoder_tokens = sample_attentions()

        decoder_text,_,cross_attention = test.test(pil_image=image)
        decoder_tokens = [c for c in decoder_text]
        encoder_tokens = ['']*cross_attention.shape[-1]

        cross_attention = (cross_attention,)
        params = head_view(cross_attention=cross_attention, encoder_tokens=encoder_tokens,
                           decoder_tokens=decoder_tokens)

        return render_template("head-view.html", params=params, base64_img=image_64_encode)
    elif request.method == "GET":
        return render_template("head-view.html", params=None, base64_img=None)


if __name__ == '__main__':
    app.run(debug=True)
