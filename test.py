import torch
from PIL import Image
from pytorch_lightning import Trainer

from src.aocr import OCR, OCRDataModule
from src.utils import dataset, utils


def test(
        test_path,
        is_dataset = False,
        output_pred_path='output.txt',
        checkpoint_path=r'lightning_logs\version_0\checkpoints\aocr-pt-epoch45-val_loss169.01.ckpt'
):

    ocr = OCR().load_from_checkpoint(checkpoint_path, output_pred_path=output_pred_path)
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

        decoder_outputs, attention_matrix = ocr(image, None, is_training=False, return_attentions=True)

        words, prob = utils.get_converted_word(decoder_outputs, get_prob=True)

        return words, prob, attention_matrix


if __name__ == "__main__":
    test(r'data\dataset\20210420_093652_rst-l8.jpg')
