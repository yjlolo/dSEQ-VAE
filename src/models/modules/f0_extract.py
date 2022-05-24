import hydra
import torch
import torchcrepe

# https://github.com/maxrmorrison/torchcrepe/blob/9c40c7737c60ae023c76466785af5e3a53b3985c/torchcrepe/core.py#L651
TIMESTAMP_HOP_SIZE = 10  # 10ms


def spec_to_f0(spec, sr, synth_audio_func, device):
    audio = torch.from_numpy(synth_audio_func(spec))
    if audio.dim() == 1: audio = audio.unsqueeze(0)

    # https://github.com/neuralaudio/hear-baseline/blob/4e097305378935928fa02094e481823e5b356d60/hearbaseline/torchcrepe.py#L135
    TIMESTAMP_HOP_SIZE_SAMPLES = (sr * TIMESTAMP_HOP_SIZE) // 1000
    ntimestamps = audio.shape[1] // TIMESTAMP_HOP_SIZE_SAMPLES + 1
    hop_size = TIMESTAMP_HOP_SIZE_SAMPLES * 1000 // sr
    timestamps = torch.tensor([i * hop_size for i in range(ntimestamps)])
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((audio.shape[0], timestamps.shape[0]))

    pitch, confidence = torchcrepe.predict(
        audio, sr, device=device, model='full', return_periodicity=True
    )

    assert pitch.shape[1] == timestamps.shape[1], \
        f"{pitch.shape[1]} vs {timestamps.shape[1]}"
    return torchcrepe.threshold.At(.5)(pitch, confidence).cpu(), timestamps
