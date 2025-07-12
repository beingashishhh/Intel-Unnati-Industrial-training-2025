from models.teacher import UNetTeacher
from models.student import UNetStudent
from models.dataset import PairedImageDataset
from torch.utils.data import DataLoader
from loss import L1_SSIM_Loss, DistillationLoss
from models.utils import eval_metrics
import torch, os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = UNetTeacher().to(device)
teacher.load_state_dict(torch.load("teacher.pth"))
teacher.eval()

student = UNetStudent().to(device)
opt = torch.optim.Adam(student.parameters(), lr=1e-4)

train_loader = DataLoader(PairedImageDataset(r"C:\Users\ashis\intal intranshep\data\Dataset\Split\train\blurry", r"C:\Users\ashis\intal intranshep\data\Dataset\Split\train\sharp"), batch_size=4, shuffle=True)
val_loader = DataLoader(PairedImageDataset(r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\blurry", r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\sharp"), batch_size=4)

loss_fn = L1_SSIM_Loss()
distill_loss = DistillationLoss()

for epoch in range(10):
    student.train()
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            teacher_out = teacher(x)
        student_out = student(x)
        loss = (0.7 * loss_fn(student_out, y) + 0.3 * distill_loss(student_out, teacher_out))
        opt.zero_grad(); loss.backward(); opt.step()

    student.eval()
    with torch.no_grad():
        val_psnr = val_ssim = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = student(x)
            p, s = eval_metrics(out, y)
            val_psnr += p; val_ssim += s
        print(f"Val → PSNR: {val_psnr/len(val_loader):.2f}, SSIM: {val_ssim/len(val_loader):.4f}")

torch.save(student.state_dict(), "student.pth")
