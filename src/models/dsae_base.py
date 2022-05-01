import logging
from abc import ABC, abstractmethod
from tkinter import E
from typing import Literal, Optional, Union, Sequence, Any, Tuple, List, Dict
import importlib

import numpy
import numpy as np
import matplotlib.pyplot as plt
import wandb
import hydra
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import Optimizer
import pytorch_lightning as pl

import src.metric as metric
from src.models.modules.audio_synth import GflFromMel
from src.utils.seq import pad_batch


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
        synth_audio: bool = False,
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
        self.synth_audio = synth_audio
        self.nake_init = nake_init

        if not self.nake_init:
            self.output_act = self._parse_dataset_param().output_act
            self.log_interval = self.hparams.logging.media_log.log_interval
        else:
            self.output_act = nn.Identity()
            self.log_interval = -1

        if self.synth_audio:
            self.audio_synth = self._configure_audioSynth()

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

    @abstractmethod
    def _run_step(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_index):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_index):
        pass
    
    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers,
              the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 
              'lr_scheduler' key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency'
              key.
            - None - Fit will run without any optimizer.
        """
        hydra.utils.log.info(f"Configuring optimizers")
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial"
        )
        if self.hparams.optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler,
                optimizer=opt
            )
            return [opt], [scheduler]
        return opt

    def _configure_audioSynth(self):
        return GflFromMel(self.hparams)
    
    def _parse_dataset_param(self):
        target_dataset_module = self.hparams.data.datasets.train._target_
        path_to_module = '.'.join(target_dataset_module.split('.')[:-1])
        dataset_module = importlib.import_module(path_to_module)
        return dataset_module

    def _resyn_audio(self, x: torch.Tensor):
        return self.audio_synth(x)

    def _log_audio(self, tag: str, audio: numpy.ndarray, stage: str):
        wandb.log({
                f"audio/{stage}-{tag}": wandb.Audio(
                    audio, sample_rate=self.audio_synth.SR, caption=tag
                )}, commit=False
        )
    
    def _log_var(self, logvar: torch.Tensor, tag: str, stage: str):
        n_plots = len(logvar)
        plt.close()
        _, ax = plt.subplots(1, n_plots, figsize=(n_plots, 1))
        for i, x in enumerate(logvar):
            avg_std = torch.sqrt(torch.exp(x)).mean(dim=0)
            if len(x.size()) == 3:
                avg_std = avg_std.mean(dim=0)
            avg_std = avg_std.cpu().detach().numpy()
            bins = list(range(len(avg_std)))
            if n_plots == 1:
                ax.bar(bins, avg_std)
                break
            ax[i].bar(bins, avg_std)

        wandb.log({stage + '_' + tag + '_std': plt})

    def _log_z_swapping(
        self,
        v: torch.Tensor,
        z: torch.Tensor,
        seq_lengths: torch.Tensor,
        label: torch.Tensor,
        stage: Literal['train', 'val'],
    ):
        label = label.cpu().detach().numpy()
        unique_labels = np.unique(label)
        chosen_idx = []
        for l in unique_labels:
            idx = np.where(label == l)[0]
            chosen_idx.append(np.random.choice(idx))
        source_idx = chosen_idx[0]
        target_idx = chosen_idx[1]
        source_v = v[source_idx].to(self.device)
        target_v = v[target_idx].to(self.device)
        source_z = z[source_idx].to(self.device)
        target_z = z[target_idx].to(self.device)
        source_seq_lengths = seq_lengths[source_idx]
        target_seq_lengths = seq_lengths[target_idx]

        source_x = self.decode(
            source_z.unsqueeze(0),
            source_v.unsqueeze(0),
            seq_lengths=source_seq_lengths.unsqueeze(0)
        )
        target_x = self.decode(
            target_z.unsqueeze(0),
            target_v.unsqueeze(0),
            seq_lengths=target_seq_lengths.unsqueeze(0)
        )
        source_swap = self.decode(
            target_z.unsqueeze(0),
            source_v.unsqueeze(0),
            seq_lengths=target_seq_lengths.unsqueeze(0)
        )
        target_swap = self.decode(
            source_z.unsqueeze(0),
            target_v.unsqueeze(0),
            seq_lengths=target_seq_lengths.unsqueeze(0)
        )

        if len(source_x['mu'].size()) == 3:
            plt.close()
            _, ax = plt.subplots(1, 4, figsize=(4, 1))
            ax[0].imshow(
                source_x['mu'][0].T.cpu().detach().numpy(),
                aspect='auto', origin='lower'
            )
            ax[1].imshow(
                source_swap['mu'][0].T.cpu().detach().numpy(),
                aspect='auto', origin='lower'
            )
            ax[2].imshow(
                target_swap['mu'][0].T.cpu().detach().numpy(),
                aspect='auto', origin='lower'
            )
            ax[3].imshow(
                target_x['mu'][0].T.cpu().detach().numpy(),
                aspect='auto', origin='lower'
            )
        elif len(source_x['mu'].size()) == 5:
            timesteps = source_x['mu'].size(1)
            h, w = source_x['mu'].size()[-2:]
            plt.close()
            _, ax = plt.subplots(4, timesteps, figsize=(timesteps * 2, 4 * 2))
            for t in range(timesteps):
                ax[0][t].imshow(
                    source_x['mu'][0][t].view(h, w, -1).cpu().detach().numpy(),
                    aspect='auto'
                )
                ax[1][t].imshow(
                    source_swap['mu'][0][t].view(h, w, -1).cpu().detach().numpy(),
                    aspect='auto'
                )
                ax[2][t].imshow(
                    target_swap['mu'][0][t].view(h, w, -1).cpu().detach().numpy(),
                    aspect='auto'
                )
                ax[3][t].imshow(
                    target_x['mu'][0][t].view(h, w, -1).cpu().detach().numpy(),
                    aspect='auto'
                )
        if self.hparams.logging.media_log.audio and self.synth_audio:
            self._log_audio(
                'source', self._resyn_audio(source_x['mu']), stage
            )
            self._log_audio(
                'source_swap', self._resyn_audio(source_swap['mu']), stage
            )
            self._log_audio(
                'target_swap', self._resyn_audio(target_swap['mu']), stage
            )
            self._log_audio(
                'target', self._resyn_audio(target_x['mu']), stage
            )

        wandb.log({stage + '_local_swap': plt})

    def _log_v_interpolation(
        self,
        v: torch.Tensor,
        z: torch.Tensor,
        seq_lengths: torch.Tensor,
        label: torch.Tensor,
        stage: Literal['train', 'val'],
        n_interpolations: int = 1
    ):
        label = label.cpu().detach().numpy()
        unique_labels = np.unique(label)
        chosen_v = []
        chosen_z = []
        chosen_seq_lengths = []
        for l in unique_labels:
            idx = np.where(label == l)[0]
            chosen_idx = np.random.choice(idx)
            chosen_v.append(v[chosen_idx])
            chosen_z.append(z[chosen_idx])
            chosen_seq_lengths.append(seq_lengths[chosen_idx])
        plt.close()
        _, ax = plt.subplots(
            1, n_interpolations + 2,
            figsize=(2 * (n_interpolations + 2) , 2 * 1)
        )
        vector = chosen_v[1] - chosen_v[0]
        """
        Plot interpolations between two end points inferred from two distinct
        classes given labels; configured as follows:

        mean_of_class_1 | .25 vector | .5 vector | .75 vector | mean_of_class_2

        The dynamical latents z_{1:T} is shared for all five reconstructions.
        """
        for i in range(0, n_interpolations + 2):
            c = chosen_v[0] + (i / (n_interpolations + 1)) * vector

            output = self.decode(
                chosen_z[0].to(self.device).unsqueeze(0),
                c.to(self.device).unsqueeze(0),
                seq_lengths=chosen_seq_lengths[0].to(self.device).unsqueeze(0)
            )
            output = output['mu']
            ax[i].imshow(
                output.detach().cpu().T,
                aspect='auto', origin='lower'
            )
            audio = self._resyn_audio(output)
            if self.hparams.logging.media_log.audio:
                self._log_audio(f'intp_{i}', audio, stage)
        wandb.log({stage + '_interpolation': plt})

    def _log_reconstruction(
        self,
        input: torch.Tensor,
        reconstruction: torch.Tensor,
        stage: Literal['train', 'val']
    ):
        input = input.cpu().detach().numpy()
        recon_np = reconstruction.cpu().detach().numpy()

        if len(input.shape) == 2:
            plt.close()
            _, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 5))
            ax[0].imshow(
                input.T, aspect='auto', origin='lower'
            )
            ax[1].imshow(
                recon_np.T, aspect='auto', origin='lower'
            )
        elif len(input.shape) == 4:
            h, w = input.shape[2:]
            timesteps = input.shape[0]
            plt.close()
            _, ax = plt.subplots(
                2, timesteps, figsize=(timesteps * 5, 2 * 5)
            )
            for t in range(timesteps):
                ax[0][t].imshow(
                    input[t].reshape(h, w, -1), aspect='auto'
                )
                ax[1][t].imshow(
                    recon_np[t].reshape(h, w, -1), aspect='auto'
                )
        wandb.log({stage + '_reconstruction': plt})

        
        if self.hparams.logging.media_log.audio and self.synth_audio:
            audio = self._resyn_audio(reconstruction.unsqueeze(0))
            self._log_audio('recon', audio, stage)
    
    def _log_v_projection(
        self,
        input: torch.Tensor,
        label: torch.Tensor,
        stage: Literal['train', 'val']
    ):
        input = input.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        pca = PCA(n_components=2).fit_transform(input)
        tsne = TSNE(n_components=2).fit_transform(input)

        plt.close()
        _, ax = plt.subplots(2, 1, figsize=(1 * 2, 2 * 2))
        for l in np.unique(label):
            idx = np.where(label == l)
            a1 = pca[idx]
            a2 = tsne[idx]
            ax[0].scatter(a1[:, 0], a1[:, 1], label=l)
            ax[1].scatter(a2[:, 0], a2[:, 1], label=l)
        wandb.log({stage + '_latent_projection': plt})
    
    def _log_generation(self, n_sample, seq_len, stage):
        outputs = self.generate(n_sample, seq_len)
        x = outputs['output']
        if len(x.size()) == 3:
            plt.close()
            _, ax = plt.subplots(1, 1, figsize=(10, 10))        
            ax.imshow(
                x.squeeze(0).T.detach().cpu(),
                aspect='auto', origin='lower'
            )
        elif len(x.size()) == 5:
            h, w = x.size()[-2:]
            plt.close()
            _, ax = plt.subplots(1, seq_len, figsize=(seq_len * 2, 1 * 2))        
            for t in range(seq_len):
                ax[t].imshow(
                    x.squeeze(0)[t].view(h, w, -1).detach().cpu(),
                    aspect='auto'
                )
        wandb.log({stage + '_generation': plt})
        
        if self.hparams.logging.media_log.audio and self.synth_audio:
            self._log_audio('generation', self._resyn_audio(x), stage)
    
    def training_step(self, batch, batch_index):
        outputs = self._run_step(batch, 'train')
        if self.current_epoch % self.log_interval == 0:
            # return self.subsample_output_dict(batch_index, outputs)
            return outputs
        else:
            return outputs['loss']

    def validation_step(self, batch, batch_index):
        outputs = self._run_step(batch, 'val')
        if self.current_epoch % self.log_interval == 0:
            # return self.subsample_output_dict(batch_index, outputs)
            return outputs
        else:
            return outputs['loss']

    def _run_step(self, batch, prefix):
        input, mask, seq_lengths, labels = batch
        outputs = self(input, seq_lengths, mask)
        outputs['ground_truth'] = labels
        losses = self.calculate_loss(outputs, outputs['mask'])
        outputs['loss'] = losses['loss']
        losses = {f'{prefix}_' + str(k): v for k, v in losses.items()}
        self.log_dict(
            losses, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        tgt_key = ['input', 'reconstruction']
        for i in tgt_key:
            outputs[i] = outputs[i].cpu()
        return outputs

    def _latent_clfr(self, v, z, y, stage):
        v = v.numpy()
        z = z.numpy()
        y = y.numpy()
        return self._iter_attr(v, z, y, stage)
    
    def _iter_attr(self, v, z, y, stage):
        if stage == 'train':
            assert not hasattr(self, 'v_clfr')
            assert not hasattr(self, 'z_clfr')
        else:
            assert hasattr(self, 'v_clfr')
            assert hasattr(self, 'z_clfr')
        data = self.hparams.data.datasets.train._target_.split('.')[-1]
        v_acc = {}
        z_acc = {}
        if data == 'Sprite':
            list_attr = ['action', 'skin', 'pant', 'top', 'hair']
            if stage == 'train':
                self.v_clfr = {}
                self.z_clfr = {}
                for k, i in zip(list_attr, range(y.shape[1])):
                    self.v_clfr[k] = metric.LdaClassifier(v, y[:, i])
                    self.z_clfr[k] = metric.LdaClassifier(z, y[:, i])
                    v_acc[k] = self.v_clfr[k](v, y[:, i], metric='acc')
                    z_acc[k] = self.z_clfr[k](z, y[:, i], metric='acc')
            else:
                for k, i in zip(list_attr, range(y.shape[1])):
                    v_acc[k] = self.v_clfr[k](v, y[:, i], metric='acc')
                    z_acc[k] = self.z_clfr[k](z, y[:, i], metric='acc')
                delattr(self, 'v_clfr')
                delattr(self, 'z_clfr')
        else:
            raise NotImplementedError
        
        return {'acc_using_global': v_acc, 'acc_using_local': z_acc}
    
    def _log_clfr_acc(self, dict_acc, prefix, stage):
        dict_acc = {
            f'{stage}_acc_{prefix}_{k}': dict_acc[k] for k in dict_acc.keys()
        }
        self.log_dict(dict_acc, on_step=False, on_epoch=True, logger=True)

    @torch.no_grad()
    def _epoch_end(self, outputs, prefix):
        seq_lengths = torch.cat([d_i['seq_lengths'] for d_i in outputs])
        max_len = int(max(seq_lengths))
        # mask = pad_batch(outputs, 'mask', max_len)
        # TODO: the target -1 has to be user input for flexibe use
        labels = torch.cat([d_i['ground_truth'][:, -1] for d_i in outputs])

        if self.hparams.logging.media_log.clfr_acc:
            z_pos = pad_batch(outputs, 'z_pos', max_len).cpu()
            local_feat = z_pos[torch.arange(z_pos.size(0)), seq_lengths-1, :]
            global_feat = torch.cat([d_i['v'] for d_i in outputs]).cpu()
            target = torch.cat([d_i['ground_truth'] for d_i in outputs]).cpu()
            clfr_acc = self._latent_clfr(
                v=global_feat, z=local_feat, y=target, stage=prefix
            )
            self._log_clfr_acc(clfr_acc['acc_using_global'], 'v', prefix)
            self._log_clfr_acc(clfr_acc['acc_using_local'], 'z', prefix)

        if self.hparams.logging.media_log.reconstruction:
            x_hat = outputs[0]['reconstruction'][0]
            x = outputs[0]['input'][0]
            self._log_reconstruction(x, x_hat, prefix)

        if self.hparams.logging.media_log.latent_var:
            v_logvar = torch.cat([d_i['v_logvar'] for d_i in outputs])
            z_pos_logvar = pad_batch(outputs, 'z_pos_logvar', max_len)
            z_prior_logvar = pad_batch(outputs, 'z_prior_logvar', max_len)
            self._log_var([v_logvar], 'v', stage=prefix)
            self._log_var([z_pos_logvar, z_prior_logvar], 'z', stage=prefix)

        if self.hparams.logging.media_log.v_project:
            v = torch.cat([d_i['v'] for d_i in outputs])
            self._log_v_projection(v, labels, prefix)

        if self.hparams.logging.media_log.generation:
            self._log_generation(1, max_len, prefix)
        
        if self.hparams.logging.media_log.v_interpolate:
            v = torch.cat([d_i['v'] for d_i in outputs])
            z_pos = pad_batch(outputs, 'z_pos', max_len)
            try:
                self._log_v_interpolation(
                    v, z_pos, seq_lengths, labels, prefix, n_interpolations=1
                )
            except IndexError as e:
                logger.warning(
                    f"{e}. Factor interpolation is not performed. "
                    "If `fast_dev_run` is set to True, "
                    "set to train/val batch size n > 1 instead. "
                    "This is due to n = 1 only contains data with a single "
                    "class of the global factor."
                )
        
        if self.hparams.logging.media_log.z_swap:
            v = torch.cat([d_i['v'] for d_i in outputs])
            z_pos = pad_batch(outputs, 'z_pos', max_len)
            try:
                self._log_z_swapping(v, z_pos, seq_lengths, labels, prefix)
            except IndexError as e:
                logger.warning(
                    f"{e}. Local latent swap is not performed. "
                    "If `fast_dev_run` is set to True, "
                    "set to train/val batch size n > 1 instead. "
                    "This is due to n = 1 only contains data with a single "
                    "class of the global factor."
                )

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.log_interval == 0:
            self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.log_interval == 0:
            self._epoch_end(outputs, 'val')


    def encode_sample_decode(self, input, sample: str, **kwargs):
        v_out = self.infer_v(
            input,
            seq_lengths=kwargs.get('seq_lengths'),
            mask=kwargs.get('mask')
        )
        assert self.z_posterior != 'x' 
        # need to refactor to include other modes
        z_pos_out = self.infer_z_pos(
            input,
            v=v_out['v'].unsqueeze(1).expand(-1, input.size(1), self.v_dim),
        )
        # recon_out = self.decode(
        #     z_pos_out['z'], v_out['v'], kwargs.get('seq_lengths')
        # )
        if sample == 'z':
            gen_out = self.generate(
                input.size(0),
                kwargs.get('seq_lengths'),
                v=v_out['v']
            )
        elif sample == 'v':
            gen_out = self.generate(
                input.size(0),
                kwargs.get('seq_lengthsh'),
                z=z_pos_out['z']
            )
        else:
            raise ValueError
        return gen_out['output']

    def subsample_output_dict(
        self, batch_index: int, x: Dict[int, torch.Tensor], n: float = 0.5
    ):
        dict_ph = {k: None for k in x.keys()}
        dict_ph['loss'] = x['loss']
        only_one_input = False
        n_data = len(x['input'])
        if int(n_data * n) <= 1:
            idx = 0
            only_one_input = True
        else:
            idx = np.random.choice(n_data, int(n_data * n), replace=False)

        # Always pick the first sample of each batch for recon visualisation
        target_keys = list(x.keys())
        if batch_index == 0:
            dict_ph['reconstruction'] = x['reconstruction'][0:1]
            dict_ph['input'] = x['input'][0:1]
        target_keys.remove('input')
        target_keys.remove('reconstruction')
        target_keys.remove('loss')  # exclude 'loss' for subsampling

        for k in target_keys:
            d_sampled = x[k][idx].detach().cpu()
            if only_one_input:
                d_sampled = d_sampled.unsqueeze(0)
            dict_ph[k] = d_sampled
        return dict_ph

    def drop_frames(self, n_frames, p=0, device=None):
        assert p >= 0 and p < 1
        if p != 0:
            n_sample = int(n_frames * p)
            return torch.randperm(n_frames, device=device)[:n_sample]
        else:
            return None

    def sample_frames(self, x, p=0):
        # TODO: consider when masking is needed
        drop_out_idx = self.drop_frames(x.size(1), p, device=x.device)
        if drop_out_idx is not None:
            mask = torch.ones_like(x, device=x.device)
            mask[:, drop_out_idx] = 0
            x = x * mask
        return x
