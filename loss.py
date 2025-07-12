import torch.nn as nn
from pytorch_msssim import ssim

class L1_SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target) + (1 - ssim(pred, target, data_range=1.0))

class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, student_out, teacher_out):
        return self.l1(student_out, teacher_out)