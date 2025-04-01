import os
import sys
import torch
import numpy as np
import math
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from . import pytorch_ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
first_run = True
class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self): pass
    def add(self):   pass
    def value(self): pass


def img_metrics(target, pred, masks=None, var=None, pixelwise=True):
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    # psnr = 20 * torch.log10(1 / (rmse + 1e-9))
    psnr = 20 * torch.log10(1 / (rmse))
    mae = torch.mean(torch.abs(target - pred))
    
    # spectral angle mapper
    mat = target * pred
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(target * target, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(pred * pred, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1))*torch.tensor(180)/math.pi)

    ssim = pytorch_ssim.ssim(target, pred)

    metric_dict = {'RMSE': rmse.cpu().numpy().item(),
                   'MAE': mae.cpu().numpy().item(),
                   'PSNR': psnr.cpu().numpy().item(),
                   'SAM': sam.cpu().numpy().item(),
                   'SSIM': ssim.cpu().numpy().item()}
    
    if masks is not None:
        # get an aggregated cloud mask over all time points and compute metrics over (non-)cloudy px
        tileTo = target.shape[1]
        mask   = torch.clamp(torch.sum(masks, dim=1, keepdim=True), 0, 1)
        mask   = mask.repeat(1, tileTo, 1, 1)
        real_B, fake_B, mask = target.cpu().numpy(), pred.cpu().numpy(), mask.cpu().numpy()

        rmse_cloudy = np.sqrt(np.nanmean(np.square(real_B[mask==1] - fake_B[mask==1])))
        rmse_cloudfree = np.sqrt(np.nanmean(np.square(real_B[mask==0] - fake_B[mask==0])))
        mae_cloudy = np.nanmean(np.abs(real_B[mask==1] - fake_B[mask==1]))
        mae_cloudfree = np.nanmean(np.abs(real_B[mask==0] - fake_B[mask==0]))
        
        metric_dict.update({
            'RMSE_cloudy': rmse_cloudy, 
            'RMSE_cloudfree': rmse_cloudfree, 
            'MAE_cloudy': mae_cloudy, 
            'MAE_cloudfree': mae_cloudfree, 
        })

    # evaluate the (optional) variance maps
    if var is not None:
        error = target - pred
        # average across the spectral dimensions
        se = torch.square(error)
        ae = torch.abs(error)

        # collect sample-wise error, AE, SE and uncertainties
 
        # define a sample as 1 image and provide image-wise statistics
        errvar_samplewise = {'error': error.nanmean().cpu().numpy().item(),
                            'mean ae': ae.nanmean().cpu().numpy().item(),
                            'mean se': se.nanmean().cpu().numpy().item(),
                            'mean var': var.nanmean().cpu().numpy().item()}
        if pixelwise:
            # define a sample as 1 multivariate pixel and provide image-wise statistics
            errvar_samplewise = {**errvar_samplewise, **{'pixelwise error': error.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise ae': ae.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise se': se.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise var': var.nanmean(0).nanmean(0).flatten().cpu().numpy()}}

        metric_dict     = {**metric_dict, **errvar_samplewise}

    if target.shape[1] == pred.shape[1] == 3: # for rgb image
        if first_run:
            loss_fn_alex.to(target.device)
        d = loss_fn_alex(target * 2 - 1, pred * 2 - 1)
        metric_dict['LPIPS'] = d.item()
    return metric_dict

class avg_img_metrics(Metric):
    def __init__(self, cloudfree_cloudy=True):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS']
        self.metrics  += ['error', 'mean se', 'mean ae', 'mean var']
        if cloudfree_cloudy:
            self.metrics += ['RMSE_cloudy', 'RMSE_cloudfree', 'MAE_cloudy', 'MAE_cloudfree']
        
        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue
            # if float("inf") == val: continue # 规避数值溢出

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics
    
   
    def extend(self,metrics):
        if isinstance(metrics,avg_img_metrics):
            for key in self.metrics:
                count1 = self.running_nonan_count[key]
                value1 = self.running_img_metrics[key]
                count2 = metrics.running_nonan_count[key]
                value2 = metrics.running_img_metrics[key]
                if count1 != 0 and count2 != 0:
                    self.running_nonan_count[key] = count1 + count2
                    self.running_img_metrics[key] = (count1 * value1 + count2 * value2) / (count1 + count2) 
                elif count1 != 0 and count2 == 0:
                    continue
                elif count1 == 0 and count2 != 0:
                    self.running_nonan_count[key] = count2
                    self.running_img_metrics[key] = value2
                else:
                    self.running_nonan_count[key] = 0
                    self.running_img_metrics[key] = np.nan
        else:
            raise TypeError