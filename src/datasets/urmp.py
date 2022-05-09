from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram

from src.datasets.constants.urmp import *
from src.datasets.preprocessor import LogCompress, TakeExp

TRANSFORM = transforms.Compose([
    MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL),
    LogCompress(),
])
OUTPUT_DENORM = TakeExp()
OUTPUT_ACT = nn.Identity()


class Urmp(Dataset):
    def __init__(
        self,
        path_to_data: str,
        instruments: List[str],
        split: str
    ):
        path_to_data = Path(path_to_data)
        assert path_to_data.exists

        audio_path = []
        audio_files = []
        labels = []
        for instrument in instruments:
            instrument_split_dir = path_to_data / instrument / split / 'audio'
            print(f"Allocating samples from {instrument_split_dir} ...")
            samples = list(instrument_split_dir.glob('*.npy'))
            print(f"Found {len(samples)} <{instrument}> samples!")
            for fpath in samples:
                audio_files.append(
                    TRANSFORM(torch.FloatTensor(np.load(str(fpath))))
                )
                instrument_label = DICT_INST_TO_IDX[fpath.stem.split('_')[3]]
                labels.append(np.array([instrument_label]))
                audio_path.append(fpath)

        self.audio_files = audio_files
        self.audio_path = audio_path
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        y = self.labels[idx]
        x = self.audio_files[idx]

        return idx, x.transpose(0, -1), x.size(-1), y
