from pathlib import Path
import numpy as np

import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.constants.dmel import *
from src.datasets.preprocessor import LogCompress, TakeExp

from data.dmelodies_dataset.dmelodies_dataset import DMelodiesDataset
from data.dmelodies_dataset.constants_latent_factors import RHYTHM_DICT


R_COND1 = [k for k, v in RHYTHM_DICT.items() if v[0] != 0]
R_COND2 = [k for k, v in RHYTHM_DICT.items() if v[-1] != 0]
TRANSFORM = transforms.Compose([
    MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL),
    LogCompress(),
])
OUTPUT_DENORM = TakeExp()
OUTPUT_ACT = nn.Identity()


class DMelodies(Dataset):
    def __init__(self, path_to_data: str):
        path_to_data = Path(path_to_data)
        assert path_to_data.exists(), f"{path_to_data} does not exist!"

        df = DMelodiesDataset()
        audio_path = []
        audio_files = []
        labels = []
        for f in path_to_data.rglob('*.wav'):
            data_idx = int(f.stem.split('_')[0])
            label_array = df._get_latents_array_for_index(data_idx)
            if label_array[3] not in R_COND1 or label_array[4] not in R_COND2:
                continue
                
            audio = torchaudio.load(f)[0].mean(0)
            norm_factor = audio.abs().max()
            audio /= norm_factor
            
            audio_path.append(f)
            audio_files.append(TRANSFORM(audio.unsqueeze(0)).squeeze(0))

            instrument_label = DICT_INST_TO_IDX[f.stem.split('-')[-1]]
            labels.append(np.append(label_array, instrument_label))

        self.audio_files = audio_files
        self.audio_path = audio_path
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        x = self.audio_files[idx]
        y = self.labels[idx]
        return idx, x.transpose(0, -1), x.size(-1), y
    
    def __repr__(self):
        return f"{self.__class__.__name__} loaded with {len(self)} samples"
