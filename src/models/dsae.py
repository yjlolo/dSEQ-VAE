import logging
from typing import List

from omegaconf import OmegaConf
import torch
import torch.nn as nn

from src.models.base import DsaeBase
from src.loss import mse_loss, kl_div
from src.models.modules import RnnEncoder, LinearLayer
from src.utils.seq import reverse_sequence


logger = logging.getLogger(__name__)


class Dsae(DsaeBase):
    def __init__(
        self,
        encoder_dims: List[int] = [64, 64],
        transition_dims: List[int] = [32, 32],
        decoder_dims: List[int] = [64, 64],
        nonlin: str = 'tanh',
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder_dims = OmegaConf.to_object(encoder_dims)
        transition_dims = OmegaConf.to_object(transition_dims)
        decoder_dims = OmegaConf.to_object(decoder_dims)
        self.encoder_dims = encoder_dims
        self.transition_dims = transition_dims
        self.decoder_dims = decoder_dims
        self.nonlin = nonlin

        """ Content encoder q(v | x_{1:T})
        """
        if self.v_posterior == '><':
            if sum([i == encoder_dims[0] for i in encoder_dims]) \
                != len(encoder_dims):
                logger.warning(f"RNN requires a same size for all layers, \
                    using the first entry of `encoder_dims` {encoder_dims[0]}."
                )
                self.encoder_dims = \
                    [encoder_dims[0] for i in range(len(encoder_dims))]
            self.net_v_encoder = RnnEncoder(
                self.input_dim,
                self.encoder_dims[0],
                'lstm',
                bidirectional=True,
                n_layer=len(self.encoder_dims),
                pool_strategy='avg',
                reverse_input=False
            )
        elif self.v_posterior == '^':
            self.net_v_encoder = LinearLayer(
                self.input_dim,
                self.encoder_dims,
                nonlin=self.nonlin,
                avg_pool=True
            )
        else:
            raise NotImplementedError
        
        self.net_v_logvar = nn.Linear(self.encoder_dims[-1], self.v_dim)
        self.net_v_mu = nn.Linear(self.encoder_dims[-1], self.v_dim)

        """ Decoder p(x_{1:T} | z_{1:T}, v)
        """
        self.input_dim_to_dec = self.z_dim + self.v_dim

        if self.likelihood == '^':
            self.net_decoder = LinearLayer(
                self.input_dim_to_dec,
                self.decoder_dims,
                nonlin=self.nonlin
            )
        elif self.likelihood == '><':
            self.net_decoder = RnnEncoder(
                self.input_dim_to_dec,
                self.decoder_dims[0],
                'lstm',
                bidirectional=True,
                n_layer=len(self.decoder_dims),
                pool_strategy=None,
                reverse_input=False
            )
        self.net_decoder_mu = nn.Sequential(
            nn.Linear(decoder_dims[-1], self.input_dim),
            self.output_act
        )

        """ Feature
        """
        if self.z_feature in ['<', '>', '><']:
            if sum([i == encoder_dims[0] for i in encoder_dims]) \
                != len(encoder_dims):
                logger.warning(f"RNN requires a same size for all layers, \
                    using the first entry of `encoder_dims` {encoder_dims[0]}."
                )
                self.encoder_dims = \
                    [encoder_dims[0] for i in range(len(encoder_dims))]
            if self.z_feature == '<':
                reverse_input = True
                bidirectional = False
            elif self.z_feature == '>':
                reverse_input = False
                bidirectional = False
            elif self.z_feature == '><':
                reverse_input = False
                bidirectional = True

            self.net_feature = RnnEncoder(
                self.input_dim,
                self.encoder_dims[0],
                'lstm',
                bidirectional=bidirectional,
                n_layer=len(self.encoder_dims),
                pool_strategy=None,
                reverse_input=reverse_input
            )
        elif self.z_feature == '^':
            self.net_feature = LinearLayer(
                self.input_dim,
                self.encoder_dims,
                nonlin=self.nonlin
            )
        else:
            raise ValueError
            
        """ Posterior q(z_t | z_{1:t-1}, x_{1:T}, v)
        """
        self.input_dim_to_posterior = self.encoder_dims[-1]
        if self.v_condition:
            self.input_dim_to_posterior += self.v_dim
        
        if self.z_posterior != 'x':
            self.input_dim_to_posterior += self.z_dim
        
        if self.z_posterior == '>>':
            if sum([i == transition_dims[0] for i in transition_dims]) \
                != len(transition_dims):
                logger.warning(
                    f"RNN requires a same size for all layers, using the first \
                        entry of `transition_dims` {transition_dims[0]}."
                )
                self.transtion_dims = \
                    [transition_dims[0] for i in range(len(transition_dims))]
            self.net_z_encoder = RnnEncoder(
                self.input_dim_to_posterior,
                self.transition_dims[0],
                'lstm',
                bidirectional=False,
                n_layer=len(self.transition_dims),
                pool_strategy=None,
                reverse_input=False
            )
            _out_dim = transition_dims[-1]
        elif self.z_posterior == '>':
            self.net_z_encoder = LinearLayer(
                self.input_dim_to_posterior,
                self.transition_dims,
                nonlin=self.nonlin
            )
            _out_dim = self.transition_dims[-1]
        elif self.z_posterior == 'x':
            self.net_z_encoder = nn.Identity()
            _out_dim = self.input_dim_to_posterior
        else:
            raise ValueError
            
        self.net_z_mu = nn.Linear(_out_dim, self.z_dim)
        self.net_z_logvar = nn.Linear(_out_dim, self.z_dim)
        
        """ Prior p(z_t | z_{1:t-1})
        """
        if self.z_prior == '>>':
            if sum([i == transition_dims[0] for i in transition_dims]) \
                != len(transition_dims):
                logger.warning(
                    f"RNN requires a same size for all layers, using the first \
                        entry of `transition_dims` {transition_dims[0]}."
                )
                self.transtion_dims = \
                    [transition_dims[0] for i in range(len(transition_dims))] 

            self.net_z_prior = RnnEncoder(
                self.z_dim,
                self.transition_dims[-1],
                'lstm',
                bidirectional=False,
                n_layer=len(self.transition_dims),
                pool_strategy=None,
                reverse_input=False
            )
            self.net_z_prior_mu = nn.Linear(self.transition_dims[-1], self.z_dim)
            self.net_z_prior_logvar = nn.Linear(self.transition_dims[-1], self.z_dim)
        elif self.z_prior == '>':
            self.net_z_prior = LinearLayer(
                self.z_dim,
                self.transition_dims,
                nonlin=self.nonlin
            )
            self.net_z_prior_mu = nn.Linear(self.transition_dims[-1], self.z_dim)
            self.net_z_prior_logvar = nn.Linear(self.transition_dims[-1], self.z_dim)
        elif self.z_prior == 'x':
            self.net_z_prior = None
            logger.warning(
                f"Prior is set to {self.z_prior}, "
                f"so the transition network is {self.net_z_prior}."
            )
        else:
            raise ValueError

        # Initialize hidden states
        self.register_buffer("z_prior_mu_0", torch.zeros(self.z_dim))
        self.register_buffer("z_prior_logvar_0", torch.zeros(self.z_dim))
        self.register_buffer("z_pos_0", torch.zeros(self.z_dim))

    def infer_v(
        self, input, deterministic=False, seq_lengths=None, mask=None
    ):
        if self.v_posterior == '><':
            assert seq_lengths is not None
            out, _, _ = self.net_v_encoder(
                input, seq_lengths=seq_lengths, mask=mask
            )
        elif self.v_posterior == '^':
            out = self.net_v_encoder(input, mask)

        mu = self.net_v_mu(out)
        logvar = self.net_v_logvar(out)

        z = self.gaussian_reparameterisation(mu, logvar, deterministic)

        return {
            'mu': mu,
            'logvar': logvar,
            'v': z
        }

    def get_feature(self, input, seq_lengths=None):
        if self.z_feature in ['<', '>', '><']:
            assert seq_lengths is not None
            out, _, _ = self.net_feature(input, seq_lengths=seq_lengths)
        elif self.z_feature == '^':
            out = self.net_feature(input)
        
        return out
    
    def infer_z_pos(
        self, input, z=None, h=None, c=None, v=None, deterministic=False,
    ):
        if self.z_posterior == '>>':
            assert z is not None
            seq_len = torch.ones(input.size(0), dtype=torch.long)
            if self.v_condition:
                assert v is not None
                input = torch.cat([input, z, v], dim=-1).unsqueeze(1)
            out, h, c = self.net_z_encoder(input, seq_len, h=h, c=c)
            out = out.squeeze(1)
        elif self.z_posterior == '>':
            if self.v_condition:
                assert v is not None
                input = torch.cat([input, z, v], dim=-1)
            out = self.net_z_encoder(input)
        elif self.z_posterior == 'x':
            if self.v_condition:
                assert v is not None
                input = torch.cat([input, v], dim=-1)
            out = self.net_z_encoder(input)
        else:
            raise NotImplementedError
        mu = self.net_z_mu(out)
        logvar = self.net_z_logvar(out)
        
        z = self.gaussian_reparameterisation(mu, logvar, deterministic)
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'h': h,
            'c': c
        }

    def infer_z_prior(
        self, input, seq_lengths=None, deterministic=False, h=None, c=None
    ):
        if self.z_prior == '>>':
            assert seq_lengths is not None
            out, h, c = self.net_z_prior(input, seq_lengths, h=h, c=c)
            mu = self.net_z_prior_mu(out)
            logvar = self.net_z_prior_logvar(out)
        elif self.z_prior == '>':
            out = self.net_z_prior(input)
            mu = self.net_z_prior_mu(out)
            logvar = self.net_z_prior_logvar(out)
            h = None
            c = None
        elif self.z_prior == '^':
            mu = torch.zeros(
                [input.size(0), input.size(1), self.z_dim],
                device=self.device
            )
            logvar = torch.zeros(
                [input.size(0), input.size(1), self.z_dim],
                device=self.device
            )
            h = None
            c = None
        z = self.gaussian_reparameterisation(mu, logvar, deterministic)
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'h': h,
            'c': c,
        }

    def decode(self, z, v, seq_lengths=None):
        v = v.unsqueeze(1).repeat(1, z.size(1), 1)

        input = torch.cat([z, v], dim=-1)
        if self.likelihood == '^':
            out = self.net_decoder(input)
        elif self.likelihood == '><':
            assert seq_lengths is not None
            out, _, _ = self.net_decoder(input, seq_lengths)
        else:
            raise NotImplementedError

        mu = self.net_decoder_mu(out)
        return {'mu': mu}

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
        }

    def generate(self, batch_size, seq_len, deterministic=False):
        z_prior_mu = self.z_prior_mu_0.expand(batch_size, self.z_dim)
        z_prior_logvar = self.z_prior_logvar_0.expand(batch_size, self.z_dim)

        z_prior_seq = \
            torch.zeros([batch_size, seq_len, self.z_dim], device=self.device)
        z_prior_mu_seq = \
            torch.zeros([batch_size, seq_len, self.z_dim], device=self.device)
        z_prior_logvar_seq = \
            torch.zeros([batch_size, seq_len, self.z_dim], device=self.device)
        output_seq = \
            torch.zeros([batch_size, seq_len, self.input_dim], device=self.device)

        # Sample the global latent variable for the sequence
        v = self.gaussian_reparameterisation(
            torch.zeros([batch_size, self.v_dim], device=self.device),
            torch.zeros([batch_size, self.v_dim], device=self.device),
            deterministic
        )
        h = None
        c = None
        z_prior = self.gaussian_reparameterisation(
            z_prior_mu, z_prior_logvar, deterministic
        )

        for t in range(seq_len):
            z_prior_mu_seq[:, t, :] = z_prior_mu
            z_prior_logvar_seq[:, t, :] = z_prior_logvar
            z_prior_seq[:, t, :] = z_prior

            z_prior_out = self.infer_z_prior(
                z_prior.unsqueeze(1), torch.ones(batch_size, dtype=torch.long),
                deterministic, h, c
            )
            z_prior_mu = z_prior_out['mu'].squeeze(1)
            z_prior_logvar = z_prior_out['logvar'].squeeze(1)
            z_prior = z_prior_out['z'].squeeze(1)
            h = z_prior_out['h']
            c = z_prior_out['c']

        output_seq = self.decode(
            z_prior_seq, v, 
            seq_lengths=seq_len * torch.ones(batch_size, device=self.device)
        )

        return {
            'output': output_seq['mu'],
            'z_mu': z_prior_mu,
            'z_logvar': z_prior_logvar,
            'z': z_prior_seq,
        }

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
        kld_v = kl_div(outputs['v_mu'], outputs['v_logvar'])

        loss = mse + kld_z + kld_v

        return {
            'loss': loss,
            'recon_loss': mse,
            'kld_z': kld_z,
            'kld_v': kld_v,
        }
