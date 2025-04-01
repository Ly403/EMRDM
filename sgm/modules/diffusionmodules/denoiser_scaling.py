from abc import ABC, abstractmethod
from typing import Tuple

import torch

class DenoiserScaling(ABC):
    @abstractmethod
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class EDMScaling:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EpsScaling:
    def __call__(
        self, sigma: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScaling:
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScalingWithEDMcNoise(DenoiserScaling):
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class ResidualEDMScaling:
    def __init__(self, sigma_input: float = 0.5, sigma_mu: float = 0.5, sigma_cov=0.9):
        self.sigma_input = sigma_input
        self.sigma_mu = sigma_mu
        self.sigma_cov = sigma_cov

    def __call__(
        self, 
        sigma: torch.Tensor, 
        st: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data = self.sigma_input
        sigma_mu_square =  (((1 - st) / st) * self.sigma_mu) ** 2
        sigma_cov = self.sigma_cov * (1 - st) / st
        c_skip = (sigma_data ** 2 + sigma_cov) / (sigma ** 2 + sigma_data ** 2 +  sigma_mu_square + 2 * sigma_cov)
        c_out = ((sigma ** 2 + sigma_mu_square) * (sigma_data ** 2) - sigma_cov ** 2) \
            / (sigma ** 2 + sigma_data ** 2 + sigma_mu_square + 2 * sigma_cov)
        c_out = c_out.sqrt()
        c_in = 1 / (sigma**2 + sigma_data**2 + sigma_mu_square + 2 * sigma_cov) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

class TemporalResidualEDMScaling:
    def __init__(self, sigma_input: float = 0.5, sigma_mu: float = 0.5, sigma_cov: float = 0.9):
        self.sigma_input = sigma_input
        self.sigma_mu = sigma_mu
        self.sigma_cov = sigma_cov

    def __call__(
        self, 
        sigma: torch.Tensor, 
        st: torch.Tensor,
        L: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data = self.sigma_input
        sigma_mu_square =  (((1 - st) / st) * self.sigma_mu) ** 2
        sigma_cov = self.sigma_cov * (1 - st) / st
        c_skip = (sigma_data ** 2 + sigma_cov) / (sigma ** 2 / L + sigma_data ** 2 +  sigma_mu_square + 2 * sigma_cov)
        c_out = ((sigma ** 2 / L + sigma_mu_square) * (sigma_data ** 2) - sigma_cov ** 2) \
            / (sigma ** 2 / L + sigma_data ** 2 + sigma_mu_square + 2 * sigma_cov)
        c_out = c_out.sqrt()
        c_in = 1 / (sigma**2 + sigma_data**2 + sigma_mu_square + 2 * sigma_cov) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise