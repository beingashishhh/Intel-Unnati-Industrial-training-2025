import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1 / mse)

def eval_metrics(pred, target):
    return psnr(pred, target).item(), ssim(pred, target, data_range=1.0).item()