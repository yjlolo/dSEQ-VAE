import hydra
from src.models.dsae import Dsae


class FreezeDsae(Dsae):
    def __init__(
        self,
        C: int = 300,
        **kwargs
    ):
        super().__init__(**kwargs)
        hydra.utils.log.info("Freezing LOCAL parameters!")
        self.switch_local(False)
        self.switched = False
        self.C = C

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
    
    def training_step(self, batch, batch_index):
        if self.current_epoch == self.C and not self.switched:
            hydra.utils.log.info("Unfreezing LOCAL parameters!")
            self.switch_local(True)
            hydra.utils.log.info("Freezing GLOBAL parameters!")
            self.switch_global(False)
            self.switched = True
            
        outputs = self._run_step(batch, 'train')
        return outputs
