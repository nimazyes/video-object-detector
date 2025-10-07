import cv2
import torch
from ultralytics import YOLO

# ===============================
#  YOLOv8 LIVE WEBCAM DETECTION
# ===============================

# --- PILIH DEVICE ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# --- LOAD MODEL ---
# Gunakan model ringan untuk real-time (yolov8n.pt)
print("[INFO] Loading YOLOv8s model...")
model = YOLO("yolov8s.pt").to(device)
model.fuse()  # Optimasi layer agar lebih cepat

# --- BUKA WEBCAM ---
cam_index = 0  # Ubah ke 1 jika webcam eksternal
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print("[ERROR] Tidak dapat membuka webcam.")
    exit()

# --- SIAPKAN PENYIMPANAN HASIL ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("webcam_detected.mp4", fourcc, fps, (width, height))

print("[INFO] Live detection dimulai — tekan 'Q' untuk berhenti.")

# --- LOOP UTAMA DETEKSI ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame agar tidak mirror
    frame = cv2.flip(frame, 1)

    # Jalankan deteksi
    results = model(frame, device=device, verbose=False)
    annotated_frame = results[0].plot()

    # Tampilkan hasil dan simpan
    cv2.imshow("YOLOv8 Webcam Detection - Tekan Q untuk keluar", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- AKHIRI PROGRAM ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("[DONE] Deteksi webcam selesai. Hasil disimpan di webcam_detected.mp4")
