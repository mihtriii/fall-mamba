
import cv2
import os

path = "/Users/mihtriii/Documents/GitHub/fall-mamba/Strategy1_Combined/train/ADL/combined_adl_0026_adl-03-cam0-rgb.mp4"
print(f"Path exists: {os.path.exists(path)}")
cap = cv2.VideoCapture(path)
print(f"Opened: {cap.isOpened()}")
if cap.isOpened():
    print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    ret, frame = cap.read()
    print(f"Read frame: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")
else:
    print("Failed to open")
