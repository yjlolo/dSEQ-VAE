import torch
import torch.nn as nn


def kl_div(mu_1, logvar_1, mu_2=None, logvar_2=None, mask=None):
    if mu_2 is None:
        mu_2 = torch.zeros_like(mu_1)
    if logvar_2 is None:
        logvar_2 = torch.zeros_like(logvar_1)

    y = 0.5 * (
        logvar_2 - logvar_1 + (
            torch.exp(logvar_1) + (mu_1 - mu_2).pow(2)
        ) / torch.exp(logvar_2) - 1
    )
    y, effect_len = masking(y, mask)

    return y.sum(dim=1).div(effect_len).mean()


def mse_loss(x_hat, x, mask=None):
    y = nn.MSELoss(reduction='none')(x_hat, x)
    y, effect_len = masking(y, mask)
    
    return y.sum(dim=1).div(effect_len).mean()


def masking(x, mask=None):
    if mask is not None:
        assert len(x.size()[1:]) == 2, \
            f"Input `x` dimension {len(x.size()[1:])} should be 2!"
        assert len(mask.size()) == 2, \
            f"Mask `mask` dimension {len(mask.size())} should be 2!"
        effect_len = mask.sum(dim=1, keepdim=True)
        x_masked = x * mask.unsqueeze(-1)
    else:
        effect_len = x.size(1) * torch.ones([x.size(0), 1], device=x.device)
        x_masked = x
    return x_masked, effect_len
