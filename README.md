The dataset is too large to upload , so the link to the google drive is given : https://drive.google.com/drive/u/0/folders/1wmC1TQLPgeZfhFmlWTiKdehJTXovAGYz
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
├── data/                          # Data directory (if used)
├── models/                        # Contains all model and utility scripts
│   ├── dataset.py                 # Loads paired blurry and sharp images
│   ├── postevaltrain_student.py  # Post-evaluation/training routines for the student
│   ├── student.py                # UNetStudent model definition
│   ├── teacher.py                # UNetTeacher model definition
│   ├── utils.py                  # Utility functions (metrics, loaders, etc.)
│   ├── video_test.py             # Tests model on webcam/video input
│   └── visualize_student.py      # Visualizes student output for analysis
├── evaluate.py                   # Script to evaluate model performance on test set
├── loss.py                       # Custom loss function definitions (L1, SSIM, etc.)
├── output_student_3...           # Output/results file (likely student logs or predictions)
├── student.pth                   # Trained student model weights
├── teacher.pth                   # Trained teacher model weights
├── train_student.py              # Training script for UNetStudent
├── train_teacher.py              # Training script for UNetTeacher
├── venv/                         # Python virtual environment directory

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


### 🖼️ Output

**Image Output:**

* A single blurry image is passed to the model.
* The sharpened result is generated as output.
* Example:

  ```
<img width="1496" height="576" alt="image" src="https://github.com/user-attachments/assets/a34dff96-118f-45e8-b592-4e4008061929" />

  ```

**Video Output:**

* An enhanced video is generated from the input provided and saved to the output directory.
* Example:

  ``` 
  Output Video    →  output_student_30fps.av  
  ```

---



