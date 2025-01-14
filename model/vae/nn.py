import torch
import torch.nn as nn


class Residual(nn.Sequential):
    """
    Wrapper to create a generic residual layer.
    """
    def forward(self, input):
        return input + super().forward(input)


# Gaussian stochastic layers
class DiagonalGaussian(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.size = latent_size

    def reparam(self, mu, logvar, random_eval=False):
        if self.training or random_eval:
            # std = exp(log(var))^0.5
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(std)
            # z = mu + std * eps
            return mu.addcmul(std, eps)
        return mu

    def sample(self, inputs, n_samples=1):
        h = self.linear(inputs)
        mu, logvar = h.unsqueeze_(1).expand(-1, n_samples, -1).chunk(2, dim=-1)
        return self.reparam(mu, logvar, random_eval=True)

    def forward(self, mu, logvar):
        return self.reparam(mu, logvar)

    def extra_repr(self):
        return 'size={}'.format(self.size)
