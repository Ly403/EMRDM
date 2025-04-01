import functools
import importlib
import os
from functools import partial
from inspect import isfunction

import fsspec
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def get_string_from_tuple(s):
    try:
        # Check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple
            t = eval(s)
            # Check if the type of t is tuple
            if type(t) == tuple:
                return t[0]
            else:
                pass
    except:
        pass
    return s


def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        lines = "\n".join(
            text_seq[start : start + nc] for start in range(0, len(text_seq), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False

    return x.ndim == 2


def isneighbors(x):
    if not isinstance(x, torch.Tensor):
        return False
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    # elif ckpt.endswith("safetensors"):
    #     sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.eval()
    return model


def get_configs_path() -> str:
    """
    Get the `configs` directory.
    For a working copy, this is the one in the root of the repository,
    but for an installed copy, it's in the `sgm` package (see pyproject.toml).
    """
    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")


def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    Will return the result of a recursive get attribute call.
    E.g.:
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    If any part of the attribute call is an integer x with current obj a, will
    try to call a[x] instead of a.x first.
    """
    attributes = attribute_path.split(".")
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    assert len(attributes) > 0, "At least one attribute should be selected"
    current_attribute = obj
    current_key = None
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])
        try:
            id_ = int(attribute)
            current_attribute = current_attribute[id_]
        except ValueError:
            current_attribute = getattr(current_attribute, attribute)

    return (current_attribute, current_key) if return_key else current_attribute



def tools_scale(tensor):
    if len(tensor.shape) == 4:
        for i in range(tensor.shape[0]):
            tensor[i,...] = (tensor[i,...] - tensor[i,...].min()) / (
                tensor[i,...].max() - tensor[i,...].min()
            )
        return tensor
    elif len(tensor.shape) == 3:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    elif len(tensor.shape) == 5:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                tensor[i,j,...] = (tensor[i,j,...] - tensor[i,j,...].min()) / (
                    tensor[i,j,...].max() - tensor[i,j,...].min()
                )
        return tensor
    else:
        raise NotImplementedError(f"Do not support this shape : {tensor.shape}")

def tools_scale2(tensor):
    return (tensor + 1.0) / 2.0 



class S1andS2_to_rgb:
    # 不做任何缩放
    def __call__(self, x):
        x = x.float()
        if x.shape[1] == 13: # S2
            x = x[:,[3,2,1],...]
        elif x.shape[1] == 2 or x.shape[1] == 15: # S1
            x = x[:,0,...].unsqueeze(dim=1)
        elif x.shape[1] == 4: # +IR
            return x[:,3,...].unsqueeze(dim=1)
        return x

class nir_to_rgb:
    # 不做任何缩放
    def __call__(self, x):
        if x.shape[1] == 4: # +IR
            return x[:,:3,...]
        else:
            return x

class identity_to_rgb:
    def __call__(self, x):
        return x.float()

# class IR_to_rgb:
#     def __call__(self, x):
#         return torch.tensor(get_rgb(x) / 255.0)
        

class get_cloud_coverage_skip_index:
    def __init__(self, key):
        self.key = key
    
    def __call__(self, batch):
        coverage = batch[self.key]
        _, index = torch.min(coverage, dim=1)
        return index
            

class scale_01_from_minus1_1:
    def __call__(self, batch_image):
        return (batch_image + 1.0) / 2.0

class sen_mtc_scale_01:
    def get_rgb(self, batch_image, ir=False):
        # # 将图像归一化到[0, 1]范围
        # out = batch_image.mul(0.5).add(0.5)
        # out = out.mul(10000).add(0.5).clamp(0, 10000).clip(0, 2000)
        # out = out - torch.min(torch.nan_to_num(out,10001))
        # out_max = torch.max(torch.nan_to_num(out,-1))
        # if out_max == 0:
        #     out = torch.ones_like(out)
        # else:
        #     out = out / out_max
        # out[torch.isnan(out)] = torch.mean(torch.nan_to_num(out,0))
        # return out
        # 将图像归一化到[0, 1]范围
        out = batch_image.mul(0.5).add(0.5)
        # 将图像缩放到[0, 10000]范围并剪切
        out = out.mul(10000).add(0.5).clamp(0, 10000)
        # 转换图像为 (batch_size, H, W, 3) 并转换为 NumPy 数组
        out = out.permute(0, 2, 3, 1) # (batch_size, H, W, 3)
        out = out.to(torch.long)

        rgb_images = []
        for image in out:
            # 提取 RGB 通道并剪切到 [0, 2000] 范围
            if ir:
                rgb = image # torch.clip(image, 0, 2000)
            else:
                r = image[:, :, 0]
                g = image[:, :, 1]
                b = image[:, :, 2]

                r = torch.clip(r, 0, 2000)
                g = torch.clip(g, 0, 2000)
                b = torch.clip(b, 0, 2000)

                # 堆叠 RGB 通道并归一化到 [0, 255] 范围
                rgb = torch.stack((r, g, b), -1)
            rgb = rgb - torch.min(torch.nan_to_num(rgb,10001))
            rgb_max = torch.max(torch.nan_to_num(rgb,-1))
            if rgb_max == 0:
                rgb = 1.0 * torch.ones_like(rgb)
            else:
                rgb = 1.0 * (rgb / rgb_max)

            # 处理 NaN 值并转换为 uint8 类型
            rgb[torch.isnan(rgb)] = torch.nanmean(rgb)
            rgb_images.append(rgb)

        # 将处理后的图像转换回 (batch_size, 3, H, W) 形状
        rgb_batch = torch.stack(rgb_images)  # (batch_size, H, W, 3)
        rgb_batch = rgb_batch.permute(0, 3, 1, 2)  # (batch_size, 3, H, W)

        return rgb_batch
    
    def __call__(self, batch_image):
        if len(batch_image.shape) == 3:
            if batch_image.shape[0] == 3:
                return self.get_rgb(batch_image.unsqueeze(0)).squeeze(0)
            else:
                return self.get_rgb(batch_image.unsqueeze(0), ir=True).squeeze(0)
        elif len(batch_image.shape) == 4:
            if batch_image.shape[1] == 3:
                return self.get_rgb(batch_image)
            else:
                return self.get_rgb(batch_image,ir=True)
        elif len(batch_image.shape) == 5:
            out = batch_image.clone()
            for i in range(batch_image.shape[1]):
                out[:,i,...] = self.get_rgb(batch_image[:,i,...])
                return out
                
    
    

            