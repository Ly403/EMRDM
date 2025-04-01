import os
import sgm
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import *
from calflops import calculate_flops
import torch
import os
os.environ["USE_COMPILE"] = "0"
os.environ["USE_FLASH_2"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,"

torch.set_float32_matmul_precision('high')

class ImageTransformerDenoiserModelFlops(ImageTransformerDenoiserModelInterface):
    def forward(self, x):
        return super().forward(x, torch.randn((1,)).cuda())

class ImageTemporalTransformerDenoiserFlops(ImageTemporalTransformerDenoiserInterface):
    def forward(self, x):
        return super().forward(x, torch.randn((1,)).cuda())

model = ImageTransformerDenoiserModelFlops(
    in_channels=28,
    out_channels=13,
    patch_size=[1,1],
    widths=[128,256,384,768],
    depths=[2,2,2,2],
    d_ffs=[256,512,768,1536],
    self_attns=[
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
        {"type": "global", "d_head": 64},
        {"type": "global", "d_head": 64},
    ],
    dropout_rate=[0.0,0.0,0.0,0.1],
    mapping_depth=2,
    mapping_width=768,
    mapping_d_ff=1536,
    mapping_dropout_rate=0.1
).cuda()

# model = ImageTemporalTransformerDenoiserFlops(
#     in_channels=7,
#     out_channels=3,
#     patch_size=[4, 4],
#     widths=[128, 256, 384],
#     depths=[2, 2, 8],
#     d_ffs=[256, 512, 768],
#     self_attns=[
#         {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
#         {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
#         {"type": "global", "d_head": 64},
#     ],
#     dropout_rate=[0.0, 0.0, 0.0],
#     mapping_depth=2,
#     mapping_width=384,
#     mapping_d_ff=768,
#     mapping_dropout_rate=0.1,
#     temporal_n_heads=16,
#     temporal_d_model=384,
#     temporal_d_k=48,
#     temporal_positional_encoding=False,
#     temporal_agg_mode="att_group",
#     temporal_dropout=0.0,
#     temporal_use_drouput=False,
#     temporal_mlp=[384, 768],
#     pad_value=None,
#     tanh=False,
# ).cuda()
print(model)

print("===========calflops============")
with torch.cuda.device(0):
    flops, macs, params = calculate_flops(
        model,
        (1, 28, 256, 256)
    )
    print(f"FLOPS: {flops}")
    print(f"MACs: {macs}")
    print(f"Params: {params}")
