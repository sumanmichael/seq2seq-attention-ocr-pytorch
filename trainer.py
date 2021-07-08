import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from aocr import OCR, OCRDataModule


class OCRLightningCLI(LightningCLI):
    defaults = {
        'model_checkpoint.monitor':'train_loss',
        'early_stopping.monitor': 'train_loss',
        'early_stopping.patience': 5,
        'data.train_list': 'data/dataset/train_list.txt',
        'data.val_list': 'data/dataset/train_list.txt',
        # 'data.num_workers': 2,
        'trainer.max_epochs': 3,
        'trainer.check_val_every_n_epoch': 5,
        'trainer.benchmark': True,
        'trainer.profiler': "pytorch",
        'trainer.log_every_n_steps': 5,
    }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.batch_size', 'model.batch_size')
        parser.link_arguments('data.img_height', 'model.img_height')
        parser.link_arguments('data.img_width', 'model.img_width')

        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        for k, v in self.defaults.items():
            parser.set_defaults({k: v})

        parser.add_optimizer_args(
            torch.optim.Adam,
            "encoder_optimizer",
            link_to='model.encoder_optimizer_args',
        )

        parser.add_optimizer_args(
            torch.optim.Adam,
            "decoder_optimizer",
            link_to='model.decoder_optimizer_args',
        )


if __name__ == "__main__":
    cli = OCRLightningCLI(OCR, OCRDataModule)
