import json
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union, List
import importlib

import numpy as np
import matplotlib.pyplot as plt
import wandb
import hydra
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import pytorch_lightning as pl

import src.metric as metric
from src.models.modules.audio_synth import GflFromMel
from src.utils.seq import pad_batch
from src.utils.util import ensure_dir


logger = logging.getLogger(__name__)


class DsaeBase(pl.LightningModule, ABC):
    def __init__(
        self,
        input_dim: Union[int, List[int]],
        z_dim: int,
        v_dim: int,
        z_feature: Literal['^', '>', '<', '><'] = '^',
        v_posterior: Literal['^', '>', '><'] = '^',
        z_posterior: Literal['x', '>', '>>'] = 'x',
        z_prior: Literal['x', '>', '>>'] = '>>',
        likelihood: Literal['^', '>', '<', '><'] = '^',
        v_condition: Optional[bool] = False,
        ar: bool = False,  # TODO: support auto-regressive
        nake_init:bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        if ar:
            logger.warning(
                f"Autoregressive decoder `ar` is set {ar} but ignored, "
                "because it is not implemented yet."
            )
            ar = False

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.v_dim = v_dim
        self.z_feature = z_feature
        self.v_posterior = v_posterior
        self.z_posterior = z_posterior
        self.z_prior = z_prior
        self.likelihood = likelihood
        self.v_condition = v_condition
        self.ar = ar
        self.nake_init = nake_init
        self.audio_synth = self._configure_audioSynth()

        if not self.nake_init:
            self.output_act = self._parse_dataset_param().OUTPUT_ACT
            self.log_interval = self.hparams.logging.media_log.log_interval
        else:
            self.output_act = nn.Identity()
            self.log_interval = -1

    def gaussian_reparameterisation(self, mu, logvar, return_mu=False):
        if return_mu:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @abstractmethod
    def infer_v(self, *args, **kwargs):
        """ q(v | x_{1:T})
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_feature(self, *args, **kwargs):
        """
        Determine how exactly should the posterior of z_{1:T}
        depends on x_{1:T}, specified by `feature`.
        The form of z_{} is later determined by `posterior`.

        feature
        '^': q(z_t | x_t, z_{})
        '>': q(z_t | x_{1:t}, z_{})
        '<': q(z_t | x_{t:T}, z_{})
        '><': q(z_t | x_{1:T}, z_{})
        """
        pass

    @abstractmethod
    def infer_z_pos(self, *args, **kwargs):
        """
        Determine how exactly should the posterior of z_{1:T} be
        factorised, specified by `posterior`.
        The form of x_{} is determined by `feature`.

        posterior
        'x': q(z_t | x_{})
        '>': q(z_t | z_{t-1}, x_{})
        '>>': q(z_t | z_{1:t-1}, x_{})
        """
        pass

    @abstractmethod
    def infer_z_prior(self, *args, **kwargs):
        """
        Determine how the prior p(z_{1:T}) is factorised, specied by `prior`.

        prior
        'x': p(z)
        '>': p(z_t | z_{t-1})
        '>>': p(z_t | z_{1:t-1})
        """
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        Determine how the likelihood p(x_{1:T} | z_{1:T}, v) is factorised,
        specified by `likelihood`.

        likelihood
        '^': p(x_t | z_t, v)
        '>': p(x_t | z_{1:t-1}, v)
        '<': p(x_t | z_{t:T}, v)
        '><': p(x_t | z_{1:T}, v)
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def configure_optimizers(self):
        hydra.utils.log.info(f"Configuring optimizers")
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial"
        )
        return opt

    def _configure_audioSynth(self):
        return GflFromMel(self.hparams)
    
    def _parse_dataset_param(self):
        target_dataset_module = self.hparams.data.datasets.train._target_
        path_to_module = '.'.join(target_dataset_module.split('.')[:-1])
        dataset_module = importlib.import_module(path_to_module)
        return dataset_module

    def _resyn_audio(self, x: torch.Tensor):
        return self.audio_synth(x.to(self.device))

    def _log_audio(self, tag: str, audio: np.ndarray, stage: str):
        wandb.log({
            f"audio/{stage}-{tag}": wandb.Audio(
                audio, sample_rate=self.audio_synth.SR, caption=tag
                )}, commit=False
        )
    
    def _log_z_swapping(
        self,
        v: torch.Tensor,
        z: torch.Tensor,
        seq_lengths: torch.Tensor,
        label: torch.Tensor,
        stage: Literal['train', 'val'],
    ):
        label = label.numpy()
        unique_labels = np.unique(label)
        chosen_idx = []
        for l in unique_labels:
            idx = np.where(label == l)[0]
            chosen_idx.append(np.random.choice(idx))
        source_idx = chosen_idx[0]
        target_idx = chosen_idx[1]
        source_v = v[source_idx:source_idx+1].to(self.device)
        target_v = v[target_idx:target_idx+1].to(self.device)
        source_z = z[source_idx:source_idx+1].to(self.device)
        target_z = z[target_idx:target_idx+1].to(self.device)
        source_seq_len = seq_lengths[source_idx:source_idx+1]
        target_seq_len = seq_lengths[target_idx:target_idx+1]

        s_x = self.decode(source_z, source_v, seq_lengths=source_seq_len)
        t_x = self.decode(target_z, target_v, seq_lengths=target_seq_len)
        s_swap = self.decode(target_z, source_v, seq_lengths=target_seq_len)
        t_swap = self.decode(source_z, target_v, seq_lengths=target_seq_len)

        n_subplot = 4
        _, ax = plt.subplots(1, n_subplot, figsize=(n_subplot, 1))
        d = {
            'source': s_x,
            'source_swap': s_swap,
            'target_swap': t_swap,
            'target': t_x
        }
        for i, k in enumerate(d.keys()):
            ax[i].imshow(
                d[k]['mu'].squeeze(0).T.cpu().numpy(),
                aspect='auto', origin='lower'
            )
            if self.hparams.logging.media_log.audio:
                self._log_audio(k, self._resyn_audio(d[k]['mu']), stage)
        wandb.log({stage + '_local_swap': plt})

    def _log_reconstruction(
        self,
        input: torch.Tensor,
        reconstruction: torch.Tensor,
        stage: Literal['train', 'val']
    ):
        input = input.numpy()
        recon_np = reconstruction.numpy()

        _, ax = plt.subplots(1, 2, figsize=(2 * 2, 1 * 2))
        ax[0].imshow(input.T, aspect='auto', origin='lower')
        ax[1].imshow(recon_np.T, aspect='auto', origin='lower')
        wandb.log({stage + '_reconstruction': plt})
        
        if self.hparams.logging.media_log.audio:
            audio = self._resyn_audio(reconstruction.unsqueeze(0))
            self._log_audio('recon', audio, stage)
    
    def _log_v_projection(
        self,
        input: torch.Tensor,
        label: torch.Tensor,
        stage: Literal['train', 'val']
    ):
        input = input.numpy()
        label = label.numpy()
        pca = PCA(n_components=2).fit_transform(input)

        _, ax = plt.subplots(1, 1, figsize=(2, 2))
        for l in np.unique(label):
            idx = np.where(label == l)
            x = pca[idx]
            ax.scatter(x[:, 0], x[:, 1], label=l)
        wandb.log({stage + '_latent_projection': plt})
    
    def _log_generation(self, n_sample, seq_len, stage):
        outputs = self.generate(n_sample, seq_len)
        x = outputs['output']
        _, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(x.squeeze(0).T.cpu(), aspect='auto', origin='lower')
        wandb.log({stage + '_generation': plt})
        
        if self.hparams.logging.media_log.audio:
            self._log_audio('generation', self._resyn_audio(x), stage)
    
    def training_step(self, batch, batch_index):
        outputs = self._run_step(batch, 'train')
        return outputs

    def validation_step(self, batch, batch_index):
        outputs = self._run_step(batch, 'val')
        return outputs

    def test_step(self, batch, batch_index, dataloader_idx):
        outputs = self._run_step(batch, 'test')
        return outputs

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.log_interval == 0:
            self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.log_interval == 0:
            self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        # We have two DLs during the eval phase,
        # hence an output list of length two
        assert len(outputs) == 2
        train_out = outputs[0]
        outputs = outputs[1]

        # Log everything on Wandb with the best ckpt
        self._epoch_end(outputs, 'test')

        # Train LDA with outputs derived from the training set
        y_train = torch.cat([d_i['ground_truth'][:, -1] for d_i in train_out])
        x = torch.cat([d_i['v_mu'] for d_i in train_out])
        clfr = {'lda': metric.LdaClassifier(x.numpy(), y_train.numpy())}

        # Conduct the evaluation derived from the val set
        seq_lengths = torch.cat([d_i['seq_lengths'] for d_i in outputs])
        max_len = int(max(seq_lengths))
        labels = torch.cat([d_i['ground_truth'][:, -1] for d_i in outputs])

        lda_score, rpa_score = metric.evaluate_latent_swap(
            model=self,
            input=torch.cat([d_i['input'] for d_i in outputs]),
            reconstruction=torch.cat(
                [d_i['reconstruction'] for d_i in outputs]).to(self.device),
            global_latent=torch.cat(
                [d_i['v_mu'] for d_i in outputs]).to(self.device),
            local_latent=torch.cat(
                [d_i['z_pos'] for d_i in outputs]).to(self.device),
            seq_lengths=torch.cat(
                [d_i['seq_lengths'] for d_i in outputs]).to(self.device),
            global_label=labels,
            mask=pad_batch(outputs, 'mask', max_len).to(self.device),
            dict_clfr=clfr
        )
        
        best_ckpt = self.trainer.checkpoint_callback.best_model_path
        run_id = best_ckpt.split('/')[-3]
        kl_reg = self.hparams.model.get('reg_weights', None)
        kl_reg = OmegaConf.to_object(kl_reg) if kl_reg is not None else None
        output_dict = {
            'seed': self.hparams.train.random_seed,
            'ckpt': best_ckpt,
            'run_id': run_id,
            'note': '',
            'optim': {
                'amsgrad': self.hparams.optim.optimizer.amsgrad
            },
            'params': {
                'model': self.hparams.model._target_.split('.')[-1],
                'data': self.hparams.data.datasets.train.path_to_data.split('/')[-2],
                'likelihood': self.likelihood,
                'z_feature': self.z_feature,
                'v_condition': self.v_condition,
                'z_dim': self.z_dim,
                'v_dim': self.v_dim,
                'kl_reg': kl_reg
            },
            'swap_local': {
                'lda': lda_score,
                'rpa': rpa_score,
            },
        }
        print(output_dict)
        if self.hparams.train.save_json:
            dataset = self.hparams.data.datasets.train._target_.split('.')[-1]
            orig_cwd = Path(hydra.utils.get_original_cwd())
            fname = "_".join([f"{k}={v}" for k, v in output_dict['params'].items()])
            fname = f'{run_id}_{fname}.json'
            tgt_dir = orig_cwd / 'benchmark' / dataset
            ensure_dir(tgt_dir)
            with open(tgt_dir / fname, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=4)

    def _run_step(self, batch, prefix):
        input, mask, seq_lengths, labels = batch
        outputs = self(input, seq_lengths, mask)
        losses = self.calculate_loss(outputs, outputs['mask'])
        outputs['loss'] = losses['loss']
        outputs['ground_truth'] = labels
        losses = {f'{prefix}_' + str(k): v for k, v in losses.items()}
        self.log_dict(
            losses, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        for k in outputs.keys():
            if k != 'loss': outputs[k] = outputs[k].cpu().detach()
        return outputs

    def _log_clfr_acc(self, dict_acc, prefix, stage):
        dict_acc = {
            f'{stage}_acc_{prefix}_{k}': dict_acc[k] for k in dict_acc.keys()
        }
        self.log_dict(dict_acc, on_step=False, on_epoch=True, logger=True)

    @torch.no_grad()
    def _epoch_end(self, outputs, prefix):
        seq_lengths = torch.cat([d_i['seq_lengths'] for d_i in outputs])
        max_len = int(max(seq_lengths))

        # NOTE: put the index -1 to the label only for convenience (instrument)
        labels = torch.cat([d_i['ground_truth'][:, -1] for d_i in outputs])

        if self.hparams.logging.media_log.reconstruction or prefix == 'test':
            x_hat = outputs[0]['reconstruction'][0]
            x = outputs[0]['input'][0]
            self._log_reconstruction(x, x_hat, prefix)

        if self.hparams.logging.media_log.v_project or prefix == 'test':
            v = torch.cat([d_i['v'] for d_i in outputs])
            self._log_v_projection(v, labels, prefix)

        if self.hparams.logging.media_log.generation or prefix == 'test':
            self._log_generation(1, max_len, prefix)
        
        if self.hparams.logging.media_log.z_swap or prefix == 'test':
            v = torch.cat([d_i['v'] for d_i in outputs])
            z_pos = pad_batch(outputs, 'z_pos', max_len)
            try:
                self._log_z_swapping(v, z_pos, seq_lengths, labels, prefix)
            except IndexError as e:
                logger.warning(
                    f"{e}. Local latent swap is not performed. "
                    "Ignore if the sanity check is performed. "
                    "If `fast_dev_run` is set to True, "
                    "set it to a larger INT instead. "
                    "This is due to the default 1 batch of data contains "
                    "only a single class of the global factor."
                )
