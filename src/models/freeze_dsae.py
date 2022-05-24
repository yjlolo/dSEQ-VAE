import hydra
import torch
from src.models.dsae import Dsae


class FreezeDsae(Dsae):
    def __init__(
        self,
        C: int = 300,
        mask_z: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        hydra.utils.log.info("Freezing LOCAL parameters!")
        self.switch_local(False)
        self.switched = False
        self.C = C
        self.mask_z = mask_z

    def switch_local(self, is_require_grad: bool):
        for p in self.net_feature.parameters():
            p.requires_grad = is_require_grad
        for p in self.net_z_encoder.parameters():
            p.requires_grad = is_require_grad
        for p in self.net_z_logvar.parameters():
            p.requires_grad = is_require_grad
        for p in self.net_z_mu.parameters():
            p.requires_grad = is_require_grad
        if self.z_prior != 'x':
            for p in self.net_z_prior.parameters():
                p.requires_grad = is_require_grad
            for p in self.net_z_prior_mu.parameters():
                p.requires_grad = is_require_grad
            for p in self.net_z_prior_logvar.parameters():
                p.requires_grad = is_require_grad

    def switch_global(self, is_require_grad: bool):
        for p in self.net_v_encoder.parameters():
            p.requires_grad = is_require_grad
        for p in self.net_v_logvar.parameters():
            p.requires_grad = is_require_grad
        for p in self.net_v_mu.parameters():
            p.requires_grad = is_require_grad
    
    def decode(self, z, v, seq_lengths=None):
        v = v.unsqueeze(1).repeat(1, z.size(1), 1)

        if self.current_epoch < self.C and self.training:
            if self.mask_z:
                z *= torch.zeros_like(z)

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

    def training_step(self, batch, batch_index):
        if self.current_epoch == self.C and not self.switched:
            hydra.utils.log.info("Unfreezing LOCAL parameters!")
            self.switch_local(True)
            hydra.utils.log.info("Freezing GLOBAL parameters!")
            self.switch_global(False)
            self.switched = True
            
        outputs = self._run_step(batch, 'train')
        return outputs
