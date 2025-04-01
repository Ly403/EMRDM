import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )

class CloudRemovalWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ):
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if c.get("concat", None) is not None:
            del c['concat']
        return self.diffusion_model(
            x,
            timesteps=t,
            **c,
            **kwargs,
        )


class TemporalCloudRemovalWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, return_attn=False, **kwargs
    ):
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
        if c.get("concat", None) is not None:
            del c['concat']
        return self.diffusion_model(
            x,
            timesteps=t,
            **c,
            return_attn=return_attn,
            **kwargs,
        )
