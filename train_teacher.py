from models.teacher import UNetTeacher
from models.dataset import PairedImageDataset
from torch.utils.data import DataLoader
from loss import L1_SSIM_Loss
from models.utils import eval_metrics
import torch, os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetTeacher().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = L1_SSIM_Loss()

train_loader = DataLoader(PairedImageDataset(r"C:\Users\ashis\intal intranshep\data\Dataset\Split\train\blurry", r"C:\Users\ashis\intal intranshep\data\Dataset\Split\train\sharp"), batch_size=4, shuffle=True)
val_loader = DataLoader(PairedImageDataset(r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\blurry", r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\sharp"), batch_size=4 ,shuffle=True)


for epoch in range(10):
    model.train()
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_psnr = val_ssim = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            p, s = eval_metrics(out, y)
            val_psnr += p; val_ssim += s
        print(f"Val → PSNR: {val_psnr/len(val_loader):.2f}, SSIM: {val_ssim/len(val_loader):.4f}")

torch.save(model.state_dict(), "teacher.pth")