import torch
from PIL import Image
from pytorch_lightning import Trainer

from aocr import OCR, OCRDataModule
from src import dataset, utils


def test(
        test_path,
        is_dataset = False,
        output_pred_path='output2.txt',
        checkpoint_path=r'C:\Users\suman\PycharmProjects\seq2seq-attention-ocr-pytorch\lightning_logs\version_0\checkpoints\epoch=4-step=114.ckpt'
):

    ocr = OCR.load_from_checkpoint(checkpoint_path, output_pred_path=output_pred_path)
    ocr.eval()

    if is_dataset:
        dm = OCRDataModule(test_list=test_path)
        t = Trainer()
        t.test(ocr, dm)
    else:
        transformer = dataset.ResizeNormalize(img_width=ocr.img_width, img_height=ocr.img_height)
        image = Image.open(test_path).convert('RGB')
        image = transformer(image)
        image = image.view(1, *image.size())
        image = torch.autograd.Variable(image)

        decoder_outputs = ocr(image, None, is_training=False)

        words, prob = utils.get_converted_word(decoder_outputs, get_prob=True)

        print("PREDICTED: ", words)
        print("WITH PROB: ", prob)

if __name__ == "__main__":
    test(r'C:\Users\suman\PycharmProjects\seq2seq-attention-ocr-pytorch\data\dataset\20210420_093652_rst-l8.jpg')
