from models.teacher import UNetTeacher
from models.student import StudentNet
from models.dataset import PairedImageDataset
from torch.utils.data import DataLoader
from models.utils import eval_metrics
import torch

teacher = UNetTeacher().cuda()
teacher.load_state_dict(torch.load("teacher.pth"))
teacher.eval()

student = StudentNet().cuda()
student.load_state_dict(torch.load("student.pth"))
student.eval()

test_loader = DataLoader(PairedImageDataset("data/test/blurry", "data/test/sharp"), batch_size=1)

print("Evaluating on test set:")

teacher_psnr = student_psnr = 0
teacher_ssim = student_ssim = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        t_out = teacher(x)
        s_out = student(x)
        p_t, s_t = eval_metrics(t_out, y)
        p_s, s_s = eval_metrics(s_out, y)
        teacher_psnr += p_t; teacher_ssim += s_t
        student_psnr += p_s; student_ssim += s_s

n = len(test_loader)
print(f"Teacher  → PSNR: {teacher_psnr/n:.2f}, SSIM: {teacher_ssim/n:.4f}")
print(f"Student  → PSNR: {student_psnr/n:.2f}, SSIM: {student_ssim/n:.4f}")
