import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI

from src.aocr import OCR, OCRDataModule


class OCRLightningCLI(LightningCLI):
    defaults = {
        # 'model_checkpoint.monitor': 'val_loss',
        # 'model_checkpoint.save_top_k': 3,
        # 'model_checkpoint.filename': 'aocr-pt-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        # 'model_checkpoint.auto_insert_metric_name': False,

        # 'early_stopping.monitor': 'val_loss',
        # 'early_stopping.patience': 5,

        'data.train_list': 'data/dataset/train_list.txt',
        'data.val_list': 'data/dataset/val_list.txt',
        'data.num_workers': 8,
        'data.batch_size': 4,

        # 'trainer.benchmark': True,
        'trainer.gpus': 1,
        'trainer.log_gpu_memory': 'all',
        # 'trainer.profiler': "pytorch",
    }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.batch_size', 'model.batch_size')
        parser.link_arguments('data.img_height', 'model.img_height')
        parser.link_arguments('data.img_width', 'model.img_width')

        parser.add_optimizer_args(torch.optim.Adadelta)
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
        # parser.add_lightning_class_args(EarlyStopping, 'early_stopping')
        # parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        for k, v in self.defaults.items():
            parser.set_defaults({k: v})


if __name__ == "__main__":
    dm = OCRDataModule(
        train_list="data/dataset/train_list.txt",
        val_list="data/dataset/val_list.txt",
        test_list=None
    )
    model = OCR()
    wandb_logger = WandbLogger(name="with_c10")
    trainer = Trainer(logger=wandb_logger)
    trainer.fit(model, dm)
