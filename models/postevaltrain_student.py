import torch
from student import UNetStudent
from dataset import PairedImageDataset
from utils import eval_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetStudent().to(device)
model.load_state_dict(torch.load("student.pth", map_location=device))
model.eval()

val_loader = torch.utils.data.DataLoader(
    PairedImageDataset(
        r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\blurry",
        r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\sharp"
    ),
    batch_size=4, shuffle=False
)

psnr_total, ssim_total = 0, 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        psnr, ssim = eval_metrics(out, y)
        psnr_total += psnr
        ssim_total += ssim

print(f"Student Model → PSNR: {psnr_total/len(val_loader):.2f}, SSIM: {ssim_total/len(val_loader):.4f}")