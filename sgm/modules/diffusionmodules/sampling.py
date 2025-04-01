"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config, tools_scale, tools_scale2
from torchvision.transforms import v2
DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class BaseResidualDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, st, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st)
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepResidualDiffusionSampler(BaseResidualDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d

# 以下EDM Sampler将sigma_t = t作为前提进行实现
class ResidualEDMSampler(SingleStepResidualDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.sigma2st = None
        
    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st
        
    def prepare_sampling_loop(self, x, mu, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)
        st0 = self.sigma2st(sigmas[0])
        # x = mu + sigmas[0] * st * x
        x = ((1 - st0) / st0) * mu + sigmas[0] * x
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0):
        st = self.sigma2st(sigma) 
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat) / st_hat - (1 - st) / st) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc)
        # d = - (x - mu) - 2 * st_hat_bc * denoised + 2 * x 
        # d = - st_hat_bc * x + mu - (denoised - x) / sigma_hat_bc
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        # d = - st_hat_bc * (x - mu)  - denoised * st_hat_bc / sigma_hat_bc + x / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            x, denoised = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
            )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.clone().detach()))
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x, others

class ResidualEulerEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step

class ResidualHeunEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            sigma_bc = append_dims(next_sigma, x.ndim)
            st = self.sigma2st(next_sigma)
            st_derivative = self.sigma2st.get_derivative_st()(next_sigma)
            
            st_bc = append_dims(st, x.ndim)
            st_derivative_bc = append_dims(st_derivative, x.ndim)
            
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, st, uc)
            d_new = (- st_derivative_bc / (st_bc ** 2)) * mu - \
                (denoised + (1 - st_bc) / st_bc * mu - x) / sigma_bc
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x

class TemporalResidualEDMSampler(ResidualEDMSampler):        
    
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            else:
                x, denoised = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others

class TemporalResidualEulerEDMSampler(TemporalResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step
  
class IdealSampler(SingleStepResidualDiffusionSampler):
    def __init__(self, input_key, mean_key, dataloader_config, *args, **kwargs):
        self.dataloader = instantiate_from_config(dataloader_config)
        self.input_key = input_key
        self.mean_key = mean_key
        super().__init__(*args, **kwargs)
    
    def normal_pdf(self, mean, scale, value):
        assert len(mean.shape) == 4
        assert len(value.shape) == 4 or len(value.shape) == 3
        assert value.shape[0] == 1 or len(value.shape) == 3
        assert mean.shape[-3:] == value.shape[-3:]
        scale_square = scale ** 2
        d = torch.tensor(value.shape).cumprod(dim=-1)[-1]
        return ((2 * torch.pi * scale_square) ** (- d / 2)) * torch.exp(((mean - value) ** 2) / (-2 * scale_square))
    
    def denoise(self, x, sigma):
        divisor = torch.zeros_like(x)
        dividend = torch.zeros_like(x)
        for data in self.dataloader.train_dataloader():
            input = data[self.input_key].to(x.device)
            mu = data[self.mean_key].to(x.device)
            y_i = input + sigma * mu
            # log_prob = torch.distributions.normal.Normal(y_i, sigma).log_prob(x)
            # prob = log_prob.exp()
            prob = self.normal_pdf(y_i, sigma, x)
            divisor += (prob * y_i).sum(dim=0)
            dividend += prob.sum(dim=0)
        return divisor / dividend
    
    def __call__(self, x, mu, num_steps=None, return_intermediate=False, return_denoised=False):
        x = x[0]
        mu = mu[0]
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        st = 1.0 /  (1.0 + sigmas[0])
        x = mu + sigmas[0] * st * x
        
        num_sigmas = len(sigmas)
        range_sigmas = self.get_sigma_gen(num_sigmas)
        # s_in = x.new_ones([x.shape[0]])
        
        intermediates = []
        denoiseds = []
        for i in range_sigmas:
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            st = 1.0 / (1.0 + sigma)
            sigma_bc = append_dims(sigma, x.ndim)
            st_bc = append_dims(st, x.ndim)
            denoised = self.denoise(x / st_bc, sigma_bc)
            
            intermediates.append(x.clone().detach().unsqueeze(0))
            denoiseds.append(denoised.clone().detach().unsqueeze(0))
            
            st = 1.0 / (1.0 + sigma)
            st = append_dims(st, x.ndim)
            d = - st * (x - mu)  - denoised * st / sigma_bc + x / sigma_bc
            dt = append_dims(next_sigma - sigma, x.ndim)
            x = self.euler_step(x, d, dt)
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x.unsqueeze(0), others       