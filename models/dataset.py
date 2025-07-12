from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class PairedImageDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        self.blur = sorted(os.listdir(blur_dir))
        self.sharp = sorted(os.listdir(sharp_dir))
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __len__(self):
        return min(len(self.blur), len(self.sharp))

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp[idx])
        if not os.path.exists(blur_path):
            raise FileNotFoundError(f"Blur image not found: {blur_path}")
        if not os.path.exists(sharp_path):
            raise FileNotFoundError(f"Sharp image not found: {sharp_path}")
        blur = self.tf(Image.open(blur_path).convert("RGB"))
        sharp = self.tf(Image.open(sharp_path).convert("RGB"))
        return blur, sharp