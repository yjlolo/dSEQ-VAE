import os
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from src.utils.util import log_hyperparameters


os.environ["WANDB_START_METHOD"] = "thread"


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    hydra.utils.log.info(f"Adding callback <EarlyStopping>")
    callbacks.append(
        EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        )
    )

    hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
    callbacks.append(
        ModelCheckpoint(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            verbose=cfg.train.model_checkpoints.verbose,
        )
    )

    hydra.utils.log.info(f"Adding callback <TQDMProgressBar>")
    callbacks.append(
        TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)
    )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop
    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        run_seed = cfg.train.random_seed
        seed_everything(run_seed)
    else:
        run_seed = None

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.num_workers = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    datamodule.run_seed = run_seed

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        logging=cfg.logging,
        data=cfg.data,
        model=cfg.model,
        train=cfg.train,
        _recursive_=False
    )

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    hydra.utils.log.info(f"Instantiating <WandbLogger>")
    wandb_logger = WandbLogger(**cfg.logging.wandb)
    hydra.utils.log.info(
        f"W&B is now watching <{cfg.logging.wandb_watch.log}>!"
    )
    wandb_logger.watch(
        model,
        log=cfg.logging.wandb_watch.log,
        log_freq=cfg.logging.wandb_watch.log_freq,
    )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(wandb_logger.experiment.dir) / "hparams.yaml").write_text(yaml_conf)

    hydra.utils.log.info(f"Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    hydra.utils.log.info("Starting evaluating!")
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    wandb_logger.experiment.finish()


@hydra.main(config_path='conf', config_name='default')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
