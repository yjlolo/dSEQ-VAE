from typing import Dict
import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def evaluate_latent_swap(
    model: pl.LightningModule,
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

    paired_global_latent = global_latent[sampled_idx]
    paired_local_latent = local_latent[sampled_idx]
    paired_seq_lengths = seq_lengths[sampled_idx]
    paired_global_label = global_label[sampled_idx]
    paired_mask = mask[sampled_idx]

    global_scores = {}
    # pre swap
    v_pre_swap = model.infer_v(
        reconstruction,
        deterministic=True,
        seq_lengths=seq_lengths, mask=mask
    )['v']
    # swap global, decode, and encode
    recon_gswap = model.decode(
        local_latent, paired_global_latent, seq_lengths=seq_lengths
    )['mu']
    v_gswap = model.infer_v(
        recon_gswap,
        deterministic=True,
        seq_lengths=seq_lengths, mask=mask
    )['v']
    # swap local, decode, and encode
    recon_lswap = model.decode(
        paired_local_latent, global_latent, seq_lengths=paired_seq_lengths
    )['mu']
    v_lswap = model.infer_v(
        recon_lswap,
        deterministic=True,
        seq_lengths=paired_seq_lengths, mask=paired_mask
    )['v']
    data = {
        'pre_swap': (v_pre_swap, global_label),
        'gswap': (v_gswap, paired_global_label),
        'lswap': (v_lswap, global_label),
    }
    for clfr_name, clfr in dict_clfr.items():
        for task, (x, y) in data.items():
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = clfr(x, y)
            global_scores['-'.join([clfr_name, task])] = acc
    
    return global_scores, global_scores.keys()


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
