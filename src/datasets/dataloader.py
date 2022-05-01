import logging
from pathlib import Path
from typing import Optional, List

import hydra
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.utils.seq import seq_collate_fn


logger = logging.getLogger(__name__)


class GeneralDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: int,
        batch_size: int,
        run_seed: Optional[int] = None
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.run_seed = run_seed

        root_path = Path(__file__).parent.parent.parent / 'data'
        datasets.train.path_to_data = str(root_path / datasets.train.path_to_data)
        datasets.val.path_to_data = str(root_path / datasets.val.path_to_data)
        self.datasets = datasets

        hydra.utils.log.info(
            f"Will be loading training data from {datasets.train.path_to_data}"
        )
        hydra.utils.log.info(
            f"Will be loading validation data from {datasets.val.path_to_data}"
        )
        

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        import contextlib
        @contextlib.contextmanager
        def temp_seed():
            torch.manual_seed(42)
            try:
                yield
            finally:
                if self.run_seed is None:
                    pass
                else:
                    seed_everything(self.run_seed)

        train_data = hydra.utils.instantiate(self.datasets.train)

        # Make sure data shuffling is aways done with the same seed.
        with temp_seed():
            self.train_dataset = torch.utils.data.Subset(
                train_data, torch.randperm(len(train_data)).tolist()
            )
        self.val_dataset = hydra.utils.instantiate(self.datasets.val)

        logger.info(f"Loaded {len(self.train_dataset)} trainig data and "
                    f"{len(self.val_dataset)} validation data."
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=seq_collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=seq_collate_fn
        )

    def test_dataloader(self) -> List[DataLoader]:
        return [self.train_dataloader(), self.val_dataloader()]
