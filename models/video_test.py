import torch
import cv2
import numpy as np
from student import UNetStudent
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetStudent().to(device)
model.load_state_dict(torch.load("student.pth", map_location=device))
model.eval()

output_video_path = "output_student_30fps.avi"
fps = 30
record_seconds = 5

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frames_to_record = fps * record_seconds
frames = []

print("Recording for 5 seconds...")
start_time = time.time()
while len(frames) < frames_to_record:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time > record_seconds:
        break

cap.release()
cv2.destroyAllWindows()
print("Recording finished. Processing frames...")

transform = lambda frame: torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

with torch.no_grad():
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(frame_rgb).unsqueeze(0).to(device)
        out = model(inp)
        out_img = out[0].cpu().permute(1, 2, 0).numpy()
        out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        # Overlay FPS text
        cv2.putText(out_bgr, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        out_vid.write(out_bgr)

out_vid.release()
print("Processed video saved as", output_video_path)