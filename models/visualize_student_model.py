import torch
from student import UNetStudent
from dataset import PairedImageDataset
from utils import eval_metrics
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetStudent().to(device)
model.load_state_dict(torch.load("student.pth", map_location=device))
model.eval()

val_loader = torch.utils.data.DataLoader(
    PairedImageDataset(
        r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\blurry",
        r"C:\Users\ashis\intal intranshep\data\Dataset\Split\val\sharp"
    ),
    batch_size=1, shuffle=True
)

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        # Convert tensors to numpy arrays for visualization
        inp_img = x[0].cpu().permute(1, 2, 0).numpy()
        out_img = out[0].cpu().permute(1, 2, 0).numpy()
        gt_img = y[0].cpu().permute(1, 2, 0).numpy()

        # Calculate SSIM score
        _, ssim_score = eval_metrics(out, y)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(inp_img)
        axs[0].set_title('Input (Blurry)')
        axs[0].axis('off')
        axs[1].imshow(out_img)
        axs[1].set_title(f'SSIM: {ssim_score:.4f}\nStudent Output')
        axs[1].axis('off')
        axs[2].imshow(gt_img)
        axs[2].set_title('Ground Truth (Sharp)')
        axs[2].axis('off')
        plt.show()
        break  # Remove this break