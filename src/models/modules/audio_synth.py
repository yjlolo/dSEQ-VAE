import importlib
import torch
from torchaudio.transforms import GriffinLim, InverseMelScale


class GflFromMel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        target_dataset_module = cfg.data.datasets.train._target_
        path_to_module = '.'.join(target_dataset_module.split('.')[:-1])
        dataset_module = importlib.import_module(path_to_module)

        self.inverse_mel = InverseMelScale(
            n_stft=dataset_module.NFFT // 2 + 1,
            n_mels=dataset_module.NMEL,
            sample_rate=dataset_module.SR,
            tolerance_loss=1e-5, max_iter=5000
        )
        self.gfl = GriffinLim(
            n_fft=dataset_module.NFFT,
            hop_length=dataset_module.HOP
        )
        
        self.SR = dataset_module.SR
        self.denorm = dataset_module.OUTPUT_DENORM
        
    def forward(self, x):
        spec = self.denorm(x).transpose(1, -1).detach()
        with torch.set_grad_enabled(True):
            stft = self.inverse_mel(spec)
            y = self.gfl(stft).squeeze(0)
        return y.cpu().numpy()
