import torch
from PIL import Image
from pytorch_lightning import Trainer

from src.aocr import OCR, OCRDataModule
from src.utils import dataset, utils


def test(
        test_path,
        is_dataset = False,
        enc_inp = None,
        output_pred_path='output.txt',
        checkpoint_path='models/aocr.pth'
):

    ocr = OCR()
    ocr.load_state_dict(torch.load(checkpoint_path))
    ocr.eval()

    if is_dataset:
        dm = OCRDataModule(test_list=test_path)
        t = Trainer()
        t.test(ocr, dm)
    else:
        if enc_inp is None:
            transformer = dataset.ResizeNormalize(img_width=ocr.img_width, img_height=ocr.img_height)
            image = Image.open(test_path).convert('L')
            image = transformer(image)
            image = image.view(1, *image.size())
            image = torch.autograd.Variable(image)
        else:
            image = enc_inp

        decoder_outputs, attention_matrix = ocr(image, None, is_training=False, return_attentions=True)

        words, prob = utils.get_converted_word(decoder_outputs, get_prob=True)

        return words, prob, attention_matrix


if __name__ == "__main__":
    w,_,_ = test(test_path=r"C:\Users\suman\PycharmProjects\seq2seq-attention-ocr-pytorch\data\predict-1-12\image.PNG")
    di = utils.digitIterator(w)
    # with open("sample.txt", "w", encoding="utf-8") as f:
    #     f.write(di.get_str())
    print(w,di.get_str())
