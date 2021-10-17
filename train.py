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

    neptune_logger = NeptuneLogger(**cfg.logger)
    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
    early_stopping = EarlyStopping(**cfg.callbacks.early_stopping)

    neptune_logger.experiment["parameters"] = OmegaConf.to_container(cfg, resolve=True)

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["logger"] = neptune_logger
    trainer_cfg["callbacks"] = [checkpoint_callback, early_stopping]
    trainer = Trainer(**trainer_cfg)
    trainer.fit(model, dm)

    neptune_logger.experiment.stop()


if __name__ == "__main__":
    main()
