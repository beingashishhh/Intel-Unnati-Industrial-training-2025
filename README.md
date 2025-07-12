# 🔍 Real-Time Image Sharpening Using Knowledge Distillation

This project develops a lightweight, real-time deep learning model that sharpens blurry video frames, improving video call quality under poor network conditions. It leverages **knowledge distillation** from a powerful **UNetTeacher** model to train a compact **UNetStudent** model suitable for low-resource devices.

## 🚀 Key Features
- Lightweight **UNetStudent** guided by **UNetTeacher**
- Real-time performance (30–60 FPS) with **SSIM ≥ 90%**
- Multi-loss training: L1, SSIM, perceptual, edge, and distillation loss
- Webcam-compatible for real-world testing
- Deployable in video calls, mobile, AR/VR, and surveillance

## 🛠 Setup
Install dependencies:
```bash
pip install torch torchvision opencv-python numpy Pillow tqdm scikit-image matplotlib
```

## 📁 Structure
```
.
├── train.py            # Trains UNetStudent using teacher output
├── test.py             # Evaluates SSIM/PSNR on test data
├── video_test.py      # Real-time sharpening via webcam
├── models/
│   ├── teacher.py # UNetTeacher architecture
│   └── student.py
 # UNetStudent architecture
├── loss_functions.py   # Custom multi-loss setup
├── dataset.py          # Loads paired blurry/sharp images
├── checkpoints/
       # Stores model weights
```

## 🧠 Usage
**Train:**
```bash
python train.py --epochs 10 --teacher_weights checkpoints/unet_teacher.pth
```
**Test:**
```bash
python test.py --model_path checkpoints/unet_student_best.pth
```
**Webcam Demo:**
```bash
python video_test.py --model_path checkpoints/unet_student_best.pth
```

## 📊 Results
- SSIM ≥ 90%
- 30–60 FPS real-time performance
- Suitable for low-power hardware

## 💼 Applications
- Video conferencing
- Surveillance video enhancement
- AR/VR visual correction
- Mobile image restoration

---
