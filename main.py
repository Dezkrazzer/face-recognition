# main.py
import matplotlib.pyplot as plt

# Import memanggil nama folder "utils"
from src.utils.image_utils import load_images_from_folder
from src.utils.core_math import EigenfaceRecognizer

# 1. Tentukan ukuran gambar standar
IMG_SIZE = (64, 64)

# 2. Muat dataset dari folder
print("Memuat dataset...")
# Karena main.py sejajar dengan folder dataset, pemanggilannya langsung nama foldernya
X_train, y_train = load_images_from_folder('src/dataset/train', target_size=IMG_SIZE)
X_test, y_test = load_images_from_folder('src/dataset/test', target_size=IMG_SIZE)

if len(X_train) == 0 or len(X_test) == 0:
    print("Error: Pastikan folder dataset/train dan dataset/test sudah diisi gambar.")
    exit()

# 3. Inisialisasi dan Latih Model Aljabar Linier
print("Menghitung Nilai Eigen dan Vektor Eigen...")
k = min(30, len(X_train)) 
recognizer = EigenfaceRecognizer(n_components=k)
recognizer.train(X_train, y_train)

# 4. Uji Pengenalan Wajah
test_img = X_test[0]
true_label = y_test[0]

predicted_label, distance, match_idx = recognizer.predict(test_img)

print(f"Hasil Prediksi : {predicted_label}")
print(f"Kenyataan      : {true_label}")
print(f"Jarak Kemiripan: {distance:.2f}")

# 5. Visualisasi Hasil
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(test_img.reshape(IMG_SIZE), cmap='gray')
axes[0].set_title(f"Gambar Uji\n(Asli: {true_label})")
axes[0].axis('off')

matched_image = X_train[match_idx]
axes[1].imshow(matched_image.reshape(IMG_SIZE), cmap='gray')
axes[1].set_title(f"Paling Mirip di Database\n(Prediksi: {predicted_label})")
axes[1].axis('off')

plt.tight_layout()
plt.show()