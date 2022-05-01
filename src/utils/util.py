from typing import Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate
import torch


STATS_KEY = 'stats'


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class LoadModel():
    def __init__(self, model_ckpt: str, cfg_path: Optional[str] = None):
        self.model_ckpt = Path(model_ckpt)
        assert self.model_ckpt.is_file()
        if isinstance(cfg_path, str):
            self.cfg_path = Path(cfg_path)
            assert self.cfg_path.is_file()
        else:
            self.cfg_path = cfg_path

    def from_hydra_yaml(self):
        self.cfg = OmegaConf.load(self.cfg_path)

        model: pl.LightningModule = instantiate(
            self.cfg.model,
            optim=self.cfg.optim,
            data=self.cfg.data,
            logging=self.cfg.logging,
            _recursive_=False,
        )
        model = model.load_from_checkpoint(self.model_ckpt, strict=True)
        model.eval()
        return model


class LoadData():
    def __init__(self, cfg_path: Optional[str] = None):
        if isinstance(cfg_path, str):
            self.cfg_path = Path(cfg_path)
            assert self.cfg_path.is_file()
        else:
            self.cfg_path = cfg_path
    
    def from_hydra_yaml(
        self,
        batch_size: Optional[int] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
    ):
        self.cfg = OmegaConf.load(self.cfg_path)
        if batch_size is not None:
            self.cfg.data.batch_size = batch_size
        if train_split is not None:
            self.cfg.data.datasets.train.split = train_split
        if val_split is not None:
            self.cfg.data.datasets.val.split = val_split
        datamodule = instantiate(self.cfg.data, _recursive_=False)
        datamodule.setup()
        return datamodule


def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None
