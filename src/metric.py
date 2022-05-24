from typing import Dict
from tqdm import tqdm
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mir_eval import melody
from src.models.modules.f0_extract import spec_to_f0


def evaluate_latent_swap(
    model: pl.LightningModule,
    input: torch.Tensor,
    reconstruction: torch.Tensor,
    global_latent: torch.Tensor,
    local_latent: torch.Tensor,
    seq_lengths: torch.Tensor,
    global_label: torch.Tensor,
    mask: torch.Tensor,
    dict_clfr: Dict,
):
    assert callable(getattr(model, 'infer_v', None))
    assert callable(getattr(model, 'decode', None))
    assert callable(getattr(model, '_resyn_audio', None))

    # Sample those with different global labels
    sampled_idx = torch.empty_like(global_label)
    for i, y in enumerate(global_label):
        candidate_idx = torch.where(global_label != y)[0]
        chosen = torch.multinomial(
            torch.ones_like(candidate_idx, dtype=float), 1
        )
        sampled_idx[i] = candidate_idx[chosen]
    sampled_idx = sampled_idx.squeeze(-1)
    assert sum(global_label[sampled_idx] != global_label) == len(global_label)

    hydra.utils.log.info("Evaluating LDA")
    global_scores = {}
    # pre swap
    v_pre_swap = model.infer_v(
        reconstruction,
        deterministic=True,
        seq_lengths=seq_lengths, mask=mask
    )['v']
    # swap global, decode, and encode
    recon_gswap = model.decode(
        local_latent,
        global_latent[sampled_idx],
        seq_lengths=seq_lengths
    )['mu']
    v_gswap = model.infer_v(
        recon_gswap,
        deterministic=True,
        seq_lengths=seq_lengths, mask=mask
    )['v']
    # swap local, decode, and encode
    recon_lswap = model.decode(
        local_latent[sampled_idx], 
        global_latent,
        seq_lengths=seq_lengths[sampled_idx]
    )['mu']
    v_lswap = model.infer_v(
        recon_lswap,
        deterministic=True,
        seq_lengths=seq_lengths[sampled_idx], mask=mask[sampled_idx]
    )['v']
    data = {
        'pre_swap': (v_pre_swap, global_label),
        'gswap': (v_gswap, global_label[sampled_idx]),
        'lswap': (v_lswap, global_label),
    }
    reconstruction = reconstruction.cpu()
    recon_lswap = recon_lswap.cpu().detach()
    recon_gswap = recon_gswap.cpu().detach()
    for clfr_name, clfr in dict_clfr.items():
        for task, (x, y) in data.items():
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = clfr(x, y)
            global_scores['-'.join([clfr_name, task])] = acc

    if not model.hparams.train.crepe_eval:
        return global_scores, None

    hydra.utils.log.info("Evaluating RPA")
    dict_rpa = {}
    bs = max(model.hparams.data.batch_size // 32, 1)
    sr = model.audio_synth.SR
    fn = model._resyn_audio
    d = model.device
    hydra.utils.log.info("Resynth audio and extract F0")
    msg = 'Ground-truth'
    f0_input, ref_t = batch_f0_extract(input, bs, sr, fn, d, msg)
    n_unpitched = torch.sum(torch.all(torch.isnan(f0_input), dim=1))
    if n_unpitched != 0:
        hydra.utils.log.warning(f"There is {n_unpitched} unpitched input seqs.")
    msg = 'Pre-swap'
    f0_recon, est_t_recon = batch_f0_extract(reconstruction, bs, sr, fn, d, msg)
    msg = 'Post-gswap'
    f0_gswap, est_t_gswap = batch_f0_extract(recon_gswap, bs, sr, fn, d, msg)
    msg = 'Post-lswap'
    f0_lswap, est_t_lswap = batch_f0_extract(recon_lswap, bs, sr, fn, d, msg)
    dict_rpa['rpa-pre_swap'] = batch_f0_eval(
        ref_t, f0_input, est_t_recon, f0_recon
    )
    dict_rpa['rpa-post_gswap'] = batch_f0_eval(
        ref_t, f0_input, est_t_gswap, f0_gswap
    )
    dict_rpa['rpa-post_lswap'] = batch_f0_eval(
        ref_t[sampled_idx], f0_input[sampled_idx], est_t_lswap, f0_lswap
    )
    return global_scores, dict_rpa


def batch_f0_extract(x, batch_size, sr, audio_syn_func, device, msg=None):
    list_x = torch.split(x, batch_size, dim=0)
    list_f0 = []
    list_t = []
    for _, x_i in tqdm(enumerate(list_x), desc=msg):
        f0, t = spec_to_f0(x_i, sr, audio_syn_func, device=device)
        list_f0.append(f0)
        list_t.append(t)
    return torch.vstack(list_f0), torch.vstack(list_t)


def batch_f0_eval(ref_time, ref_freq, est_time, est_freq):
    rpa = []
    for rt, rf, et, ef in zip(ref_time, ref_freq, est_time, est_freq):
        rpa.append(f0_eval(rt.numpy(), rf.numpy(), et.numpy(), ef.numpy()))
    return np.mean(rpa)


def f0_eval(ref_time, ref_freq, est_time, est_freq):
    ref_v, ref_c, est_v, est_c = melody.to_cent_voicing(
        ref_time, ref_freq, est_time, est_freq
    )
    rpa = melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    return rpa
    

class LdaClassifier():
    def __init__(self, X, Y):
        self.lda = LinearDiscriminantAnalysis()
        self._train(X, Y)

    def _train(self, X, Y):
        self.lda.fit(X, Y)

    def __call__(self, x, y, metric='f1'):
        out = self.predict(x)
        if metric == 'f1':
            return f1_score(y, out, average='macro')
        elif metric == 'acc':
            return accuracy_score(y, out)

    def predict(self, x):
        return self.lda.predict(x)
