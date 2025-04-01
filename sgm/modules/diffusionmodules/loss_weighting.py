from abc import ABC, abstractmethod

import torch


class DiffusionLossWeighting(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0

class ResidualEDMWeighting:
    def __init__(self, sigma_input: float = 0.5, sigma_mu: float = 0.5, sigma_cov=0.9):
        self.sigma_input = sigma_input
        self.sigma_mu = sigma_mu
        self.sigma_cov = sigma_cov

    def __call__(
        self, 
        sigma: torch.Tensor,
        st: torch.Tensor
    ) -> torch.Tensor:
        # sigma_data = (self.sigma_input ** 2 + (((1 - st) / st) ** 2) * (self.sigma_mu ** 2)).sqrt()
        sigma_data = self.sigma_input
        sigma_mu_square =  (((1 - st) / st) * self.sigma_mu) ** 2
        sigma_cov = self.sigma_cov * (1 - st) / st
        return (sigma ** 2 + sigma_data ** 2 + sigma_mu_square + 2 * sigma_cov) / ((sigma ** 2 + sigma_mu_square) * (sigma_data ** 2) - sigma_cov ** 2)
        # return (1. / sigma ** 2)

class ResidualSoftMinSnrWeighting:
    def __init__(self, gamma=5):
        self.gamma = gamma

    def __call__(
        self, 
        sigma: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return 1.0 / (sigma ** 2 + 1.0 / self.gamma)
        # return (1. / sigma ** 2)

class TemporalResidualEDMWeighting:
    def __init__(self, sigma_input: float = 0.5, sigma_mu: float = 0.5, sigma_cov: float = 0.9):
        self.sigma_input = sigma_input
        self.sigma_mu = sigma_mu
        self.sigma_cov = sigma_cov

    def __call__(
        self, 
        sigma: torch.Tensor,
        st: torch.Tensor,
        L: int
    ) -> torch.Tensor:
        sigma_data = self.sigma_input
        sigma_mu_square =  (((1 - st) / st) * self.sigma_mu) ** 2
        sigma_cov = self.sigma_cov * (1 - st) / st
        return (sigma ** 2 / L + sigma_data ** 2 + sigma_mu_square + 2 * sigma_cov) / ((sigma ** 2 / L + sigma_mu_square) * (sigma_data ** 2) - sigma_cov ** 2)