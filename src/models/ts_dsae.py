from typing import List
from copy import deepcopy

import hydra
import torch

from src.models.freeze_dsae import FreezeDsae
from src.utils.seq import reverse_sequence
from src.loss import mse_loss, kl_div


class TsDsae(FreezeDsae):
    def __init__(
        self,
        reg_weights: List[float] = [1.0, 1.0, 1.0, 1.0],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reg_weights = reg_weights
        self.global_encoder_snapshot = False
        
    def infer_v(
        self, input, deterministic=False, seq_lengths=None, mask=None,
        global_encoder=None, global_mu=None, global_logvar=None
    ):
        if global_encoder is None:
            global_encoder = self.net_v_encoder
            global_mu = self.net_v_mu
            global_logvar = self.net_v_logvar

        if self.v_posterior == '><':
            assert seq_lengths is not None
            out, _, _ = global_encoder(
                input, seq_lengths=seq_lengths, mask=mask
            )
        elif self.v_posterior == '^':
            out = global_encoder(input, mask)

        mu = global_mu(out)
        logvar = global_logvar(out)

        z = self.gaussian_reparameterisation(mu, logvar, deterministic)

        return {
            'mu': mu,
            'logvar': logvar,
            'v': z
        }
    
    def forward(
        self, input, seq_lengths, mask=None, deterministic=False
    ):
        T_max = input.size(1)
        batch_size = input.size(0)

        """Encode feature
        """
        if self.z_feature == '<':
            input = reverse_sequence(input, seq_lengths)

        x = self.get_feature(input, seq_lengths)

        # Configure intitial states
        z_pos_0 = self.z_pos_0.expand(batch_size, self.z_dim)
        z_prior_mu_0 = self.z_prior_mu_0.expand(batch_size, self.z_dim)
        z_prior_logvar_0 = self.z_prior_logvar_0.expand(batch_size, self.z_dim)
        z_prev = z_pos_0

        """Encode v
        """
        v_out = self.infer_v(
            input, deterministic, seq_lengths=seq_lengths, mask=mask
        )
        if self.current_epoch >= self.C:
            if not self.global_encoder_snapshot:
                self.net_v_prior = deepcopy(self.net_v_encoder)
                self.net_v_mu_prior = deepcopy(self.net_v_mu)
                self.net_v_logvar_prior = deepcopy(self.net_v_logvar)
                self.global_encoder_snapshot = True
            v_prior_out = self.infer_v(
                input, deterministic, seq_lengths=seq_lengths, mask=mask,
                global_encoder=self.net_v_prior,
                global_mu=self.net_v_mu_prior,
                global_logvar=self.net_v_logvar_prior
            )
            v_prior_mu = v_prior_out['mu']
            v_prior_logvar = v_prior_out['logvar']
        else:
            v_prior_mu = torch.zeros_like(v_out['mu'])
            v_prior_logvar = torch.zeros_like(v_out['logvar'])

        # Create placeholders
        z_pos_mu_seq = torch.zeros(
            [batch_size, T_max, self.z_dim], device=self.device
        )
        z_pos_logvar_seq = torch.zeros(
            [batch_size, T_max, self.z_dim], device=self.device
        )
        z_pos_seq = torch.zeros(
            [batch_size, T_max, self.z_dim], device=self.device
        )
        h = None
        c = None

        if self.z_posterior != 'x':
            for t in range(T_max):
                """Posterior q(z_t|z_{<t}, x, v)
                """
                z_pos_out = self.infer_z_pos(
                    x[:, t, :],
                    z_prev,
                    h,
                    c,
                    v_out['v'],
                    deterministic
                )
                z_prev = z_pos_out['z']
                h = z_pos_out['h']
                c = z_pos_out['c']

                z_pos_mu_seq[:, t, :] = z_pos_out['mu']
                z_pos_logvar_seq[:, t, :] = z_pos_out['logvar']
                z_pos_seq[:, t, :] = z_pos_out['z']
        else:
            z_pos_out = self.infer_z_pos(
                x, v=v_out['v'].unsqueeze(1).repeat(1, T_max, 1),
                deterministic=deterministic
            )
            z_pos_mu_seq = z_pos_out['mu']
            z_pos_logvar_seq = z_pos_out['logvar']
            z_pos_seq = z_pos_out['z']

        
        z_prior_out = self.infer_z_prior(z_pos_seq, seq_lengths, deterministic)
        z_prior_mu_seq = torch.cat([
            z_prior_mu_0.unsqueeze(1),
            z_prior_out['mu'][:, :-1, :]
        ], dim=1)
        z_prior_logvar_seq = torch.cat([
            z_prior_logvar_0.unsqueeze(1),
            z_prior_out['logvar'][:, :-1, :]
        ], dim=1)

        z_prior_0 = self.gaussian_reparameterisation(
            z_prior_mu_0, z_prior_logvar_0, deterministic
        )
        z_prior_seq = torch.cat([
            z_prior_0.unsqueeze(1),
            z_prior_out['z'][:, :-1, :]
        ], dim=1)
            
        """Decode p(x_t|z, v)
        """
        x_out = self.decode(z_pos_seq, v_out['v'], seq_lengths=seq_lengths)

        if self.z_feature == '<':
            input = reverse_sequence(input, seq_lengths)

        return {
            'reconstruction': x_out['mu'],
            'input': input,
            'mask': mask,
            'seq_lengths': seq_lengths,
            'z_pos': z_pos_seq,
            'z_pos_mu': z_pos_mu_seq,
            'z_pos_logvar': z_pos_logvar_seq,
            'z_prior': z_prior_seq,
            'z_prior_mu': z_prior_mu_seq,
            'z_prior_logvar': z_prior_logvar_seq,
            'v': v_out['v'],
            'v_mu': v_out['mu'],
            'v_logvar': v_out['logvar'],
            'v_prior_mu': v_prior_mu,
            'v_prior_logvar': v_prior_logvar,
        }

    def random_swap_decode_encode(self, z, v, seq_lengths, mask, swap):
        idx = torch.arange(len(z), out=torch.LongTensor())
        idx = random_permute(idx.to(self.device))
        _, sorted_seq_length_indices = torch.sort(seq_lengths[idx])
        sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
        sorted_seq_lengths = seq_lengths[idx][sorted_seq_length_indices]

        if swap == 'local':
            v = v[sorted_seq_length_indices]
            z = z[idx][sorted_seq_length_indices]
        elif swap == 'global':
            v = v[idx][sorted_seq_length_indices]
            z = z[sorted_seq_length_indices]
        else:
            raise ValueError
        
        seq_lengths = seq_lengths[idx][sorted_seq_length_indices]
        mask = mask[idx][sorted_seq_length_indices]
        recon = self.decode(z, v, seq_lengths)['mu']
        out = self(recon, sorted_seq_lengths, mask)
        v_mu = out['v_mu']
        v_logvar = out['v_logvar']
        z_mu = out['z_pos_mu']
        z_logvar = out['z_pos_logvar']

        return (v_mu, v_logvar), (z_mu, z_logvar), \
            (sorted_seq_length_indices, idx[sorted_seq_length_indices])

    def calculate_loss(self, outputs, mask=None):
        mse = mse_loss(
            outputs['reconstruction'],
            outputs['input'],
            mask
        )
        kld_z = kl_div(
            outputs['z_pos_mu'], outputs['z_pos_logvar'],
            outputs['z_prior_mu'], outputs['z_prior_logvar'],
            mask
        )
        kld_v = kl_div(
            outputs['v_mu'], outputs['v_logvar'],
            outputs['v_prior_mu'], outputs['v_prior_logvar'],
        )
        if self.current_epoch >= self.C:
            (v_gswap), (z_gswap), idx_gswap = self.random_swap_decode_encode(
                outputs['z_pos'], outputs['v'], outputs['seq_lengths'],
                mask, swap='global'
            )
            (v_lswap), (z_lswap), idx_lswap = self.random_swap_decode_encode(
                outputs['z_pos'], outputs['v'], outputs['seq_lengths'],
                mask, swap='local'
            )
            kld_v_gswap = self.reg_weights[0] * kl_div(
                *v_gswap,
                outputs['v_mu'][idx_gswap[1]],
                outputs['v_logvar'][idx_gswap[1]]
            )
            kld_z_gswap = self.reg_weights[1] * kl_div(
                *z_gswap,
                outputs['z_pos_mu'][idx_lswap[0]],
                outputs['z_pos_logvar'][idx_lswap[0]],
                mask[idx_lswap[0]]
            )
            kld_v_lswap = self.reg_weights[2] * kl_div(
                *v_lswap,
                outputs['v_mu'][idx_gswap[0]],
                outputs['v_logvar'][idx_gswap[0]]
            )
            kld_z_lswap = self.reg_weights[3] * kl_div(
                *z_lswap,
                outputs['z_pos_mu'][idx_lswap[1]],
                outputs['z_pos_logvar'][idx_lswap[1]],
                mask[idx_lswap[1]]
            )
        else:
            kld_v_gswap = torch.zeros_like(kld_v)
            kld_z_gswap = torch.zeros_like(kld_v)
            kld_v_lswap = torch.zeros_like(kld_v)
            kld_z_lswap = torch.zeros_like(kld_v)

        loss = mse + kld_z + kld_v + \
            kld_v_gswap + kld_z_gswap + \
            kld_v_lswap + kld_z_lswap

        return {
            'loss': loss,
            'recon_loss': mse,
            'kld_z': kld_z,
            'kld_v': kld_v,
            'kld_v_gswap': kld_v_gswap,
            'kld_z_gswap': kld_z_gswap,
            'kld_v_lswap': kld_v_lswap,
            'kld_z_lswap': kld_z_lswap,
        }

    def training_step(self, batch, batch_index):
        if self.current_epoch == self.C and not self.switched:
            hydra.utils.log.info("Unfreezing LOCAL parameters!")
            self.switch_local(True)
            self.switched = True

        outputs = self._run_step(batch, 'train')
        return outputs

def random_permute(x):
    return torch.multinomial(torch.ones_like(x, dtype=float), len(x))
