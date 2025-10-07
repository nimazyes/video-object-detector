# Real-Time Object Detection using YOLOv8s

Proyek ini menggunakan model YOLOv8s dari Ultralytics untuk melakukan deteksi objek secara langsung melalui webcam.  
Model ini dapat mengenali hingga 80 kelas objek dari dataset COCO, seperti manusia, mobil, anjing, kucing, sepeda, dan lainnya.

---

## 1. Kriteria dan Kemampuan

Model YOLOv8s mampu mendeteksi berbagai objek umum dari dataset COCO, di antaranya:
- person
- car, bus, truck, bicycle, motorcycle
- cat, dog, horse, cow, sheep
- bottle, cup, bowl, chair, sofa, tv, laptop, cellphone, book, dll.

Jumlah total kelas: **80**.

---

## 2. Persyaratan Sistem

- Python 3.8 atau lebih baru  
- GPU NVIDIA (opsional, tetapi disarankan untuk kinerja cepat)  
- Kamera (webcam)  
- Koneksi internet (saat pertama kali mengunduh model YOLO)

---

## 3. Instalasi

Jalankan perintah berikut satu per satu di terminal (Command Prompt, PowerShell, atau VSCode Terminal):

```bash
# 1. Install PyTorch dengan dukungan GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Jika tidak memiliki GPU, gunakan perintah ini:
# pip install torch torchvision torchaudio

# 2. Install pustaka tambahan
pip install opencv-python
pip install tqdm
pip install matplotlib
pip install ultralytics
pip install pandas
pip install seaborn

## 3. Cara Menjalankan

- Pastikan semua pustaka sudah terinstal dengan benar.
- Simpan file utama dengan nama yolo_webcam.py di dalam folder proyek.
- Jalankan perintah berikut di terminal untuk memulai deteksi:

python yolo_webcam.py

- Kamera akan otomatis aktif dan model akan mulai mendeteksi objek secara langsung.
- Tekan tombol Q di jendela tampilan untuk keluar dari program.
