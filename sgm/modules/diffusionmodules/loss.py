from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config, tools_scale, tools_scale2
from .denoiser import Denoiser
from .sigma2st import Sigma2St
from torchvision.transforms import v2

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

class ResidualDiffusionLoss(StandardDiffusionLoss):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, sigma2st, input, mu, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w)
    
    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + mu + noise * sigmas_bc
        return noised_input

class TemporalResidualDiffusionLoss(ResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        # 这里有两种实现方法，一种是三个时间点用不一样的noise，一种是三个时间点用一样的noise
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w)
    