import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from src.aocr import OCR, OCRDataModule


class OCRLightningCLI(LightningCLI):
    defaults = {
        'model_checkpoint.monitor': 'val_loss',
        'model_checkpoint.save_top_k': 3,
        'model_checkpoint.filename': 'aocr-pt-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        'model_checkpoint.auto_insert_metric_name': False,

        # 'early_stopping.monitor': 'val_loss',
        # 'early_stopping.patience': 5,

        'data.train_list': 'data/dataset/train_list.txt',
        'data.val_list': 'data/dataset/val_list.txt',
        'data.num_workers': 2,
        'data.batch_size' : 4,
        # 'trainer.max_epochs': 2,
        # 'trainer.check_val_every_n_epoch': 1,
        'trainer.benchmark': False,
        'trainer.gpus' : 1,
        # 'trainer.profiler': "pytorch",
        # 'trainer.log_every_n_steps': 50,
        'trainer.num_sanity_val_steps':0
    }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.batch_size', 'model.batch_size')
        parser.link_arguments('data.img_height', 'model.img_height')
        parser.link_arguments('data.img_width', 'model.img_width')

        parser.add_optimizer_args(torch.optim.Adadelta)
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
        # parser.add_lightning_class_args(EarlyStopping, 'early_stopping')
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        for k, v in self.defaults.items():
            parser.set_defaults({k: v})


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    cli = OCRLightningCLI(OCR, OCRDataModule)
