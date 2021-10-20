import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from src.aocr import OCR, OCRDataModule

import dotenv; dotenv.load_dotenv()



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = OCRDataModule(**cfg.data)
    model = OCR(optim_kwargs=cfg.optim, **cfg.model)

    callbacks = []
    if cfg.callbacks.checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["callbacks"] = callbacks

    neptune_logger = None
    if cfg.logger:
        neptune_logger = NeptuneLogger(**cfg.logger)
        neptune_logger.experiment["parameters"] = OmegaConf.to_container(cfg, resolve=True)
        trainer_cfg["logger"] = neptune_logger

    trainer = Trainer(**trainer_cfg)
    trainer.fit(model, dm)

    if neptune_logger:
        neptune_logger.experiment.stop()


if __name__ == "__main__":
    main()
