import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img, tools_scale, tools_scale2, append_dims)
# from ..modules.learning.metrics import img_metrics, avg_img_metrics
import os
import numpy as np
from PIL import Image
import rasterio
import pandas as pd
import cv2
from matplotlib import pyplot as plt  
from mpl_toolkits.axes_grid1 import ImageGrid
class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        # elif path.endswith("safetensors"):
        #     sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log

class ResidualDiffusionEngine(DiffusionEngine):
    def __init__(self, sigma_st_config, to_rgb_config, scale_01_config=None, ideal_sampler_config = None, mean_key="mu", use_flash_attn2 = False, compile_model = False, image_metrics="metrics", count_train_time=False, count_sample_time=False,*args, **kwargs):
        if compile_model:
            os.environ["USE_COMPILE"] = "1"
        else:
            os.environ["USE_COMPILE"] = "0"
        if use_flash_attn2:
            os.environ["USE_FLASH_2"] = "1"
        else:
            os.environ["USE_FLASH_2"] = "0"
        
        # 两种数据集的评价指标计算方式不一样，这里通过一个不大优美的方法来区分开
        assert image_metrics in ["metrics", "evaluator"], "image_metrics should be either metrics or evaluator"
        if image_metrics == "metrics":
            from sgm.modules.learning.metrics import img_metrics, avg_img_metrics
            self.img_metrics = img_metrics
            self.avg_metrics = avg_img_metrics()
        elif image_metrics == "evaluator":
            from sgm.modules.learning.evaluator import img_metrics, avg_img_metrics
            self.img_metrics = img_metrics
            self.avg_metrics = avg_img_metrics()
        super().__init__(compile_model=compile_model, *args, **kwargs)
        self.mean_key = mean_key
        self.sigma2st = instantiate_from_config(sigma_st_config)
        self.scale_01 = instantiate_from_config(
            default(scale_01_config, {"target": "sgm.util.scale_01_from_minus1_1"})
        )
        if ideal_sampler_config is not None:
            self.ideal_sampler = instantiate_from_config(ideal_sampler_config)
        else:
            self.ideal_sampler = None
        # assert self.sampler.has("set_sigma2st"), "The sampler does not have set_sigma2st function, maybe you should use the residual sampler."
        try:
            self.sampler.set_sigma2st(self.sigma2st)
        except:
            raise NotImplementedError("The sampler does not have set_sigma2st function, maybe you should use the residual sampler.")
        self.to_rgb_func = instantiate_from_config(to_rgb_config)
        self.count_sample_time = count_sample_time
        self.count_train_time = count_train_time
        if self.count_train_time:
            self.train_start_event = None
            self.train_end_event = None
            self.train_time = 0.0
            self.train_count = 0
    
    def on_train_batch_start(self, *args, **kwargs):
        super().on_train_batch_start(*args, **kwargs)
        if self.count_train_time:
            self.train_start_event = torch.cuda.Event(enable_timing=True)
            self.train_end_event = torch.cuda.Event(enable_timing=True)
            self.train_start_event.record()
    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        if self.count_train_time:
            self.train_end_event.record()
            torch.cuda.synchronize()
            train_time = self.train_start_event.elapsed_time(self.train_end_event)
            self.train_count += 1
            self.train_time += train_time
            avg_train_time = self.train_time / self.train_count
            self.log_dict({"train_time": avg_train_time}, sync_dist=True, on_step=True, on_epoch=False)
            # print(f"Train time: {avg_train_time} per batch.")      
    
    def get_input(self, batch, key):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[key]
    
    def forward(self, x, mu, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, self.sigma2st, x, mu, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict
    
    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch, self.input_key)
        x = self.encode_first_stage(x)
        mu = self.get_input(batch, self.mean_key)
        mu = self.encode_first_stage(mu)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, mu, batch)
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        mu: torch.Tensor,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
        ideal_sample=False, 
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c, st: self.denoiser(
            self.model, input, sigma, c, st, **kwargs
        )
        if ideal_sample:
            samples = self.ideal_sampler(randn, mu, return_intermediate=return_intermediate, return_denoised=return_denoised)
        else:
            samples = self.sampler(denoiser, randn, mu, cond, uc=uc, return_intermediate=return_intermediate, return_denoised=return_denoised)
        return samples
    
    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:
                        # image cond
                        xc = x[:n,...]
                    else:
                        xc = x
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = self.to_rgb_func(xc)
        return log
    
    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
        return_add_mu: bool = False,
        return_add_noise: bool = False,
        return_cond: bool = False,
        return_reconstrcution: bool = False,
        return_ideal_samples: bool = False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        mu = mu.to(self.device)[:N]
        log["inputs"] = self.to_rgb_func(x.clone().detach())
        log["mean"] = self.to_rgb_func(mu.clone().detach())
        
        if return_add_mu or return_add_noise:
            sigmas = self.sampler.discretization(
                self.sampler.num_steps, device=self.device
            )
            mus = [tools_scale(x.clone().detach())] if return_add_mu else None
            noises = [tools_scale(x.clone().detach())] if return_add_noise else None
            
            for i in reversed(self.sampler.get_sigma_gen(self.sampler.num_steps)):
                sigma = sigmas[i]
                st = self.sigma2st(sigma)
                if return_add_mu:
                    _ = x + (1 - st) / st * mu
                    mus.append(tools_scale(_.detach()))
                if return_add_noise:
                    _ = x + (1 - st) / st * mu + torch.randn_like(x) * sigma
                    noises.append(tools_scale(_.detach()))
            
            if return_add_mu:
                log["mu_shifting"] = self._get_denoise_row_from_list(mus, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
            if return_add_noise:
                log["mu_noise_shifting"] = self._get_denoise_row_from_list(noises, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
            
        z = self.encode_first_stage(x)
        z_mu = self.encode_first_stage(mu)
        if return_reconstrcution:
            log["reconstructions"] = self.to_rgb_func(self.decode_first_stage(z.clone().detach()))
        if return_cond:
            log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
        
        if sample:
            with self.ema_scope("Plotting"):
                samples, others = self.sample(
                    c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate, return_denoised=return_denoised, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = self.to_rgb_func(samples)
            if return_intermediate:
                log["intermediate"] = self._get_denoise_row_from_list(others['intermediates'], to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
                
            if return_denoised:
                log["denoised"] = self._get_denoise_row_from_list(others['denoiseds'], to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
        
        # if return_ideal_samples:
        #     assert self.ideal_sampler is not None, "Ideal sampler is not defined"
        #     with self.ema_scope("Plotting"):
        #         samples, others = self.sample(
        #             c, z_mu, shape=z.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate, return_denoised=return_denoised, ideal_sample=True, **sampling_kwargs
        #         )
        #     samples = self.decode_first_stage(samples)
        #     log["ideal_samples"] = samples
        #     if return_intermediate:
        #         log["ideal_intermediate"] = self._get_denoise_row_from_list(others['intermediates']) * 2.0 - 1.0
                
        #     if return_denoised:
        #         log["ideal_denoised"] = self._get_denoise_row_from_list(others['denoiseds']) * 2.0 - 1.0
  
        return log
    
    def _get_denoise_row_from_list(self, samples, desc='', to_rgb_func=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        if to_rgb_func != None:
            denoise_grid = to_rgb_func(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def shared_test_step(self, batch):
        target = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        sampling_kwargs = {}
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        if self.count_sample_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)
        if self.count_sample_time:
            end_event.record()
            torch.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event)
            self.log_dict({"sample_time": sample_time}, sync_dist=True, on_step=True, on_epoch=False)
        
        for i in range(samples.shape[0]):
            _target = target[i,...]
            _samples = samples[i,...]
            _target = self.scale_01(_target)
            _samples = self.scale_01(_samples)
            metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            self.log_dict(metrics, sync_dist=True, batch_size=1, on_epoch=True)
            _mu = self.scale_01(mu[i,...])
            raw_metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_mu.unsqueeze(0))
            raw_metrics = {"raw_" + k:v for k, v in raw_metrics.items()}
            self.log_dict(raw_metrics, sync_dist=True, batch_size=1, on_epoch=True)
            self.avg_metrics.add(metrics)
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        self.shared_test_step(batch=batch)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.shared_test_step(batch=batch)

    # def on_train_start(self):
    #     # flops, params = thop.profile(self.model.diffusion_model, inputs=(torch.randn([1,28,256,256],device=self.device),\
    #     #     torch.randn(1,device=self.device),))
    #     # flops, params = thop.clever_format([flops, params], "%.3f")
    #     # print(flops, params)
    #     model_fwd = lambda: self.model.diffusion_model(torch.randn([1,28,256,256],device=self.device),\
    #         torch.randn(1,device=self.device))
    #     fwd_flops = measure_flops(self.model.diffusion_model,model_fwd)
    #     print(fwd_flops)
    
    @torch.no_grad()
    def on_predict_epoch_start(self, *args, **kwargs):
        self.all_pred_metrics = []

    @torch.no_grad()
    def on_predict_epoch_end(self, *args, **kwargs):
        metrics = {}
        for metric in self.all_pred_metrics:
            for k,v in metric.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        
        pd.DataFrame(metrics).to_csv(self.logger.save_dir + "/metrics.csv")

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        mu = self.get_input(batch, self.mean_key)
        assert mu.shape[0] == 1, "batch size should be 1."
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        sampling_kwargs = {}
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)

        path = self.logger.save_dir + "/sample/"
        os.makedirs(path, exist_ok=True)
        image_path = self.get_input(batch, "image_path")[0]
        image_path = image_path.split("/")[-1]
        _, image_path_extension  = os.path.splitext(image_path)
        target = self.get_input(batch, self.input_key)
        target_path = image_path.replace(image_path_extension, "_target.png")
        target = self.scale_01(target)
        samples = self.scale_01(samples)
        target_rgb = np.moveaxis((self.to_rgb_func(target)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
        Image.fromarray(target_rgb).save(path + target_path)
        rgb_path = image_path.replace(image_path_extension, ".png")
        rgb = np.moveaxis((self.to_rgb_func(samples)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
        Image.fromarray(rgb).save(path +  rgb_path)
        sample = samples[0].cpu().numpy()
        if image_path_extension == ".tif":
            with rasterio.open(path + image_path, 'w', driver='GTiff', height=sample.shape[1], width=sample.shape[2], count=sample.shape[0], dtype=sample.dtype) as dst:
                dst.write(sample)
        mu_path = image_path.replace(image_path_extension, "_mu.png")
        mu = self.scale_01(mu)
        mu_rgb = np.moveaxis((self.to_rgb_func(mu)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
        Image.fromarray(mu_rgb).save(path + mu_path)
        # calculate the metrics
        metrics = self.img_metrics(target=target, pred=samples)
        metrics["image_path"] = image_path
        self.all_pred_metrics.append(metrics)
    
    @torch.no_grad()
    def on_test_epoch_start(self, *args, **kwargs):
        self.avg_metrics.reset()

    @torch.no_grad()
    def on_test_epoch_end(self, *args, **kwargs):
        avg_metrics = self.avg_metrics.value()
        final_metrics = {}
        for k,v in avg_metrics.items():
            final_metrics["final_" + k] = v
        self.log_dict(final_metrics, sync_dist=True, on_epoch=True)
        
    @torch.no_grad()
    def on_validation_epoch_start(self, *args, **kwargs):
        self.avg_metrics.reset()
    
    @torch.no_grad()
    def on_validation_epoch_end(self, *args, **kwargs):
        avg_metrics = self.avg_metrics.value()
        final_metrics = {}
        for k,v in avg_metrics.items():
            final_metrics["final_" + k] = v
        self.log_dict(final_metrics, sync_dist=True, on_epoch=True)
        

class TemporalResidualDiffusionEngine(ResidualDiffusionEngine):
    
    def __init__(self, mask_key=None, image_path_key="image_path", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self.image_path_key = image_path_key

    def _get_temporal_denoise_row_from_list(self, samples, desc='', to_rgb_func=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b t c h w -> b t n c h w')
        denoise_grid = rearrange(denoise_grid, 'b t n c h w -> (b t n) c h w')
        if to_rgb_func != None:
            denoise_grid = to_rgb_func(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
    
    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
        return_add_mu: bool = False,
        return_add_noise: bool = False,
        return_cond: bool = False,
        return_reconstrcution: bool = False,
        return_mask: bool = False,
        return_attn: bool = False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        if return_mask:
            mask = self.get_input(batch, self.mask_key)
            if mask is not None:
                for i in range(mask.shape[1]):
                    log[f"mask_timestep{i}"] = mask[:N,i,...].unsqueeze(dim=1) * 2.0 - 1.0

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )
        
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        mu = mu.to(self.device)[:N]
        log["inputs"] = self.to_rgb_func(x.clone().detach())
        for i in range(mu.shape[1]):
            log[f"mean_timestep{i}"] = self.to_rgb_func(mu[:,i,...].clone().detach())

        if return_add_mu or return_add_noise:
            sigmas = self.sampler.discretization(
                self.sampler.num_steps, device=self.device
            )
            for index in range(mu.shape[1]):
                _mu = mu[:,index,...] 
                mus = [tools_scale(x.clone().detach())] if return_add_mu else None
                noises = [tools_scale(x.clone().detach())] if return_add_noise else None

                for i in reversed(self.sampler.get_sigma_gen(self.sampler.num_steps)):
                    sigma = sigmas[i]
                    st = self.sigma2st(sigma)
                    if return_add_mu:
                        _ = x + (1 - st) / st * _mu
                        mus.append(tools_scale(_.detach()))
                    if return_add_noise:
                        _ = x + (1 - st) / st * _mu + torch.randn_like(x) * sigma
                        noises.append(tools_scale(_.detach()))
                
                if return_add_mu:
                    log[f"mu_shifting_timestep{index}"] = self._get_denoise_row_from_list(mus, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
                if return_add_noise:
                    log[f"mu_noise_shifting_timestep{index}"] = self._get_denoise_row_from_list(noises, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
                
        z = self.encode_first_stage(x)
        z_mu = self.encode_first_stage(mu)
        if return_reconstrcution:
            log["reconstructions"] = self.to_rgb_func(self.decode_first_stage(z.clone().detach()))
        if return_cond:
            log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
        
        if sample:
            sampling_kwargs = {}
            with self.ema_scope("Plotting"):
                samples, others = self.sample(
                    c, z_mu, batch, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_intermediate=return_intermediate, return_denoised=return_denoised, return_attn=return_attn, **sampling_kwargs
                )
                samples = self.decode_first_stage(samples)
            log["samples"] = self.to_rgb_func(self.scale_01(samples) * 2.0 - 1.0)
            # log['samples'] = self.to_rgb_func(samples)
            if return_intermediate:
                log["intermediate"] = self._get_temporal_denoise_row_from_list(others['intermediates'], to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
                
            if return_denoised:
                log["denoised"] = self._get_denoise_row_from_list(others['denoiseds'], to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
            
            if return_attn:
                attns = others['attns']
                # n_heads, batch_size, t, h, w
                # 只看最后一步的attn
                attn = attns[-1]
                attn = attn.view(-1, attn.shape[2], 1, attn.shape[3], attn.shape[4])
                log["attn"] = self._get_denoise_row_from_list(attn, to_rgb_func=self.to_rgb_func) * 2.0 - 1.0
    
        return log
    
    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:
                        # image cond
                        xc = x[:n,...]
                    elif x.dim() == 5:
                        # a list of images
                        xc = [x[:,i,...] for i in range(x.shape[1])]
                    else:
                        xc = x
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                if isinstance(xc,list):
                    for i, _xc in enumerate(xc):
                        log[embedder.input_key + f"_timestep{i}"] = self.to_rgb_func(_xc)
                else:
                    log[embedder.input_key] = self.to_rgb_func(xc)
        return log

    
    @torch.no_grad()
    def shared_test_step(self, batch):
        target = self.get_input(batch, self.input_key)
        mu = self.get_input(batch, self.mean_key)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )
        sampling_kwargs = {}
        
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        if self.count_sample_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with self.ema_scope("Plotting"):
            samples, _ = self.sample(
                c, z_mu, batch, shape=z_mu.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)
        if self.count_sample_time:
            end_event.record()
            torch.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event)
            self.log_dict({"sample_time": sample_time}, sync_dist=True, on_step=True, on_epoch=False)
        
        for i in range(samples.shape[0]):
            _target = target[i,...]
            _samples = samples[i,...]
            _target = self.scale_01(_target)
            _samples = self.scale_01(_samples)
            metrics = self.img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            self.log_dict(metrics, sync_dist=True, batch_size=1, on_epoch=True)
            self.avg_metrics.add(metrics)
        # raw_metrics = img_metrics(target=target, pred=mu)
        # raw_metrics = {"raw_" + k:v for k, v in raw_metrics.items()}
        # self.log_dict(raw_metrics, sync_dist=True)
    
    
    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        mu: torch.Tensor,
        batch: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
        return_attn: bool = False,
        ideal_sample=False,
        **kwargs,
    ):
        sampling_kwargs = {k: self.get_input(batch, k) for k in self.loss_fn.batch2model_keys}
        # skip_index = self.loss_fn.get_skip_index(batch)
        randn = torch.randn(batch_size, *shape).to(self.device)
        denoiser = lambda input, sigma, c, st, return_attn=False: self.denoiser(
            self.model, input, sigma, c, st, return_attn, **sampling_kwargs, **kwargs
        )
        samples = self.sampler(denoiser, randn, mu, cond, uc=uc, return_intermediate=return_intermediate, return_denoised=return_denoised, return_attn=return_attn)
        return samples
    

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # 这个函数是没处理多卡多进程的问题的，只能支持rank=0
        mu = self.get_input(batch, self.mean_key)
        assert mu.shape[0] == 1, "batch size should be 1."
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[]
        )

        sampling_kwargs = {}
        z_mu = self.encode_first_stage(mu)
        N = z_mu.shape[0]
        with self.ema_scope("Plotting"):
            samples, others = self.sample(
                c, z_mu, batch, shape=z_mu.shape[1:], uc=uc, batch_size=N, return_attn=True, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)

        attns = others["attns"]
        attn = attns[-1]
        # 只拿最后一个头的attn
        attn = attn.view(-1, attn.shape[2], 1, attn.shape[3], attn.shape[4])[0] 
        
        path = self.logger.save_dir + "/sample/"
        os.makedirs(path, exist_ok=True)
        image_path = self.get_input(batch, self.image_path_key)[0]
        if isinstance(image_path, list): # 应对多帧输入
            image_path = image_path[0]
        image_path = image_path.split("/")[-1]
        _, image_path_extension = os.path.splitext(image_path)
        target = self.get_input(batch, self.input_key)
        target_path = image_path.replace(image_path_extension, "_target.png")
        target = self.scale_01(target)
        samples = self.scale_01(samples)
        target_rgb = np.moveaxis((self.to_rgb_func(target)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
        Image.fromarray(target_rgb).save(path + target_path)
        rgb_path = image_path.replace(image_path_extension, ".png")
        rgb = np.moveaxis((self.to_rgb_func(samples)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
        Image.fromarray(rgb).save(path +  rgb_path)
        sample = samples[0].cpu().numpy()
        if image_path_extension == ".tif":
            with rasterio.open(path + image_path, 'w', driver='GTiff', height=sample.shape[1], width=sample.shape[2], count=sample.shape[0], dtype=sample.dtype) as dst:
                dst.write(sample)
        # calculate the metrics
        # fig, ax = plt.subplots(2, mu.shape[1], figsize=(0.03 * mu.shape[3], 0.019 * mu.shape[4]))  # 2行3列 
        # attn_rgb_full_path = path + image_path.replace(image_path_extension, "_attn.png")
        # im = None
        for i in range(mu.shape[1]):
            mu_rgb = self.scale_01(mu[:,i])
            mu_rgb_path = image_path.replace(image_path_extension, f"_timestep{i}.png")
            mu_rgb = np.moveaxis((self.to_rgb_func(mu_rgb)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
            Image.fromarray(mu_rgb).save(path + mu_rgb_path)
            
            c_rgb_path = image_path.replace(image_path_extension, f"_timestep{i}_cond.png")
            cond = self.scale_01(c["concat"][:,i])
            c_rgb = np.moveaxis((self.to_rgb_func(cond)[0] * 255).cpu().numpy().astype(np.uint8),0,-1)
            Image.fromarray(c_rgb.squeeze(-1), mode="L").save(path + c_rgb_path)
            
        #     attn_rgb = attn[i]
        #     attn_rgb = (attn_rgb - attn_rgb.min()) / (attn_rgb.max() - attn_rgb.min())
        #     # bilinear上采样到和mu一样的大小
        #     attn_rgb = torch.nn.functional.interpolate(
        #         attn_rgb.unsqueeze(0), 
        #         size=mu_rgb.shape[:2], 
        #         mode="bilinear", 
        #         align_corners=False
        #     )[0]
        #     attn_rgb_path = image_path.replace(image_path_extension, f"_timestep{i}_attn.png")
        #     attn_rgb = np.moveaxis((attn_rgb * 255).cpu().numpy().astype(np.uint8),0, -1)
        #     attn_heat_map = cv2.applyColorMap(attn_rgb, cv2.COLORMAP_JET)
        #     cv2.imwrite(path + attn_rgb_path, attn_heat_map)
        #     ax[0][i].axis("off")
        #     ax[1][i].axis("off")
        #     ax[0][i].imshow(mu_rgb, alpha=1)
        #     im = ax[1][i].imshow(attn_rgb, alpha=1.0, cmap="viridis")
        
        # plt.subplots_adjust(wspace=0.01)  # 设置子图之间的间隔为0.01
        # plt.subplots_adjust(hspace=0.01)  # 减小子图的纵向间距
        
        # # Add colorbar
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03)
        # cbar.set_ticks(np.linspace(0, 255, 11))
        # cbar.set_ticklabels([f'{i/255:.1f}' for i in np.linspace(0, 255, 11)])
        # cbar.ax.tick_params(labelsize=5)  # 设置色标刻度字体大小
        
        # plt.show() 
        # plt.savefig(attn_rgb_full_path.replace(".png", ".svg"), format="svg", dpi=1200, bbox_inches='tight',)
        # plt.close()  
        metrics = self.img_metrics(target=target, pred=samples)
        metrics["image_path"] = image_path
        self.all_pred_metrics.append(metrics)
        
    
