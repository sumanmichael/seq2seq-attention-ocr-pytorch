import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.aocr import OCR, OCRDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = OCRDataModule(**cfg.data)
    model = OCR(optim_kwargs=cfg.optim, **cfg.model)

    wandb_logger = WandbLogger(config=OmegaConf.to_container(cfg, resolve=True), **cfg.logger)
    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
    early_stopping = EarlyStopping(**cfg.callbacks.early_stopping)

    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, dm)



if __name__ == "__main__":
    main()
