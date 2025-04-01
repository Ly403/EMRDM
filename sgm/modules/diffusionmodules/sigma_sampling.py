import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
    
class UniformSampling:
    def __init__(self, sigma_min=0.02, sigma_max=5.0):
        self.log_sigma_min = torch.log(torch.tensor(sigma_min))
        self.log_sigma_max = torch.log(torch.tensor(sigma_max))

    def __call__(self, n_samples, rand=None):
        log_sigma = (self.log_sigma_max - self.log_sigma_min) * default(rand, torch.rand((n_samples,))) + self.log_sigma_min
        return log_sigma.exp()


class ResidualEDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, p_max=80.0):
        self.p_mean = p_mean
        self.p_std = p_std
        self.p_max = torch.tensor(p_max)

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        sigma = log_sigma.exp()
        sigma = torch.clamp(input=sigma,max=self.p_max)
        return sigma

class TestEDMSampling:
    def __init__(self, sigma):
        self.sigma = torch.tensor(sigma)
        
    def __call__(self, n_samples, rand=None):
        return torch.ones((n_samples,)) * self.sigma
