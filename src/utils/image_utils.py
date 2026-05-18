# image_utils.py
import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, target_size=(64, 64)):
    images = []
    labels = []
    
    # Membaca setiap file dalam folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Mengambil nama sebelum '_' sebagai label identitas
            label = filename.split('_')[0]
            img_path = os.path.join(folder_path, filename)
            
            # Buka gambar, ubah ke Grayscale ('L'), dan resize
            img = Image.open(img_path).convert('L')
            img = img.resize(target_size)
            
            # Ubah matriks gambar (64x64) menjadi array 1D (4096,)
            img_array = np.array(img).flatten()
            
            images.append(img_array)
            labels.append(label)
            
    return np.array(images), np.array(labels)