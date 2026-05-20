import os
import io
import cv2
import numpy as np
import torch
from src.utils.eigen import manual_eig

IMG_SIZE = (50, 50) 
UI_IMAGE_SIZE = (150, 150)
MAX_CHUNK_SIZE = 95 * 1024 * 1024  # Batasan limit 95MB per file

# Konfigurasi Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "src", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

torch.set_float32_matmul_precision('medium') 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset_cli(dataset_path, num_components=50):
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Folder dataset '{dataset_path}' tidak ditemukan!")
        return

    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, file))

    images, labels, saved_ui_images = [], [], []
    total_files = len(all_files)
    
    print(f"\n🚀 [CLI] Memulai ekstraksi {total_files} gambar menggunakan Haar Cascade...")
    
    for idx, img_path in enumerate(all_files):
        img = cv2.imread(img_path)
        if img is None: continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            gray_crop = gray[y:y+h, x:x+w]
            color_crop = img[y:y+h, x:x+w]
        else:
            continue  # Mengabaikan file yang terlewat (miss) oleh Haar Cascade

        resized_gray = cv2.resize(gray_crop, IMG_SIZE)
        images.append(resized_gray.flatten())
        
        resized_color = cv2.resize(color_crop, UI_IMAGE_SIZE)
        saved_ui_images.append(cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB))
        labels.append(os.path.basename(os.path.dirname(img_path))) 
        
        if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
            print(f" -> Progress: {idx + 1} / {total_files} gambar diproses...")
                
    Gamma_np = np.array(images).T
    if Gamma_np.shape[1] == 0:
        print("❌ Error: Tidak ada wajah yang berhasil diekstraksi.")
        return

    M = Gamma_np.shape[1]
    print(f"\n⚡ [VRAM] Menghitung Aljabar Linear di {DEVICE.type.upper()}...")
    Gamma = torch.tensor(Gamma_np, dtype=torch.float32, device=DEVICE)
    mean_face = torch.mean(Gamma, dim=1, keepdim=True)
    Phi = Gamma - mean_face
    C = torch.matmul(Phi.T, Phi) / M
    
    eigenvalues, eigenvectors = manual_eig(C, num_components=num_components)
    
    U = torch.matmul(Phi, eigenvectors)
    U = U / torch.norm(U, dim=0, keepdim=True)
    projected_faces = torch.matmul(U.T, Phi)
    
    # Bungkus seluruh model ke dalam objek dictionary sebelum dipecah
    checkpoint = {
        'mean_face': mean_face.cpu(),
        'eigenfaces': U.cpu(),
        'projected_training_faces': projected_faces.cpu(),
        'dataset_labels': labels,
        'dataset_images': saved_ui_images 
    }
    
    print("\n📦 [Sharding] Menyimpan dan memecah model menjadi partisi biner @95MB...")
    
    # Serialize checkpoint ke memory buffer terlebih dahulu
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    raw_bytes = buffer.getvalue()
    total_bytes = len(raw_bytes)
    
    # Bersihkan file lama berawalan "Eigen_Weight_" agar tidak bentrok
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("Eigen_Weight_"):
            os.remove(os.path.join(OUTPUT_DIR, f))
            
    # Pecah byte menjadi beberapa chunk file
    part_number = 1
    start_idx = 0
    
    while start_idx < total_bytes:
        end_idx = min(start_idx + MAX_CHUNK_SIZE, total_bytes)
        chunk_data = raw_bytes[start_idx:end_idx]
        
        file_name = f"Eigen_Weight_{part_number:04d}.pt"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        with open(file_path, "wb") as f_out:
            f_out.write(chunk_data)
            
        print(f" -> Berhasil menulis: {file_name} ({len(chunk_data) / (1024*1024):.2f} MB)")
        
        part_number += 1
        start_idx = end_idx

    print(f"\n✅ Selesai! Seluruh model berhasil di-shard ke dalam folder: src/dataset/\n")

if __name__ == "__main__":
    print("=== EIGENFACE DATASET BUILDER (CLI SHARDING VERSION) ===")
    path_input = input("Masukkan absolut path folder dataset citra mentah:\n> ").strip()
    # Bersihkan pembungkus tanda petik jika ada hasil drag-and-drop folder
    if path_input.startswith(('"', "'")) and path_input.endswith(('"', "'")):
        path_input = path_input[1:-1]
        
    build_dataset_cli(path_input)