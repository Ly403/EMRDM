import torch

class UnitWeighting:
    def __call__(self, sigma):
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2.0

class ResidualEDMWeighting:
    def __init__(self, sigma_input: float = 0.5, sigma_mu: float = 0.5):
        self.sigma_input = sigma_input
        self.sigma_mu = sigma_mu

    def __call__(
        self, 
        sigma: torch.Tensor,
        st: torch.Tensor
    ) -> torch.Tensor:
        sigma_data = (self.sigma_input ** 2 + (((1 - st) / st) ** 2) * (self.sigma_mu ** 2)).sqrt()
        return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2