# src/main_gui.py
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from eigen import calculate_manual_eigen, manual_euclidean_distance

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition - FATISDA UNS")
        self.root.geometry("800x500")
        
        # Variabel State
        self.dataset_folder = ""
        self.test_image_path = ""
        self.IMG_SIZE = (64, 64)
        
        # Model Data
        self.mean_face = None
        self.eigenfaces = None
        self.weights_train = None
        self.y_train = []
        self.X_train_images = []
        
        self.setup_ui()

    def setup_ui(self):
        # ================= KIRI: KONTROL [cite: 89-94] =================
        left_frame = tk.Frame(self.root, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        tk.Label(left_frame, text="Insert Your Dataset", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        btn_dataset = tk.Button(left_frame, text="Choose Folder", command=self.load_dataset)
        btn_dataset.pack(anchor="w")
        self.lbl_dataset_status = tk.Label(left_frame, text="No Folder Chosen")
        self.lbl_dataset_status.pack(anchor="w", pady=(0, 20))
        
        tk.Label(left_frame, text="Insert Your Image", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        btn_image = tk.Button(left_frame, text="Choose File", command=self.load_test_image)
        btn_image.pack(anchor="w")
        self.lbl_image_status = tk.Label(left_frame, text="No File Chosen")
        self.lbl_image_status.pack(anchor="w", pady=(0, 20))
        
        tk.Label(left_frame, text="Result", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.lbl_result = tk.Label(left_frame, text="None", fg="green", font=("Arial", 12, "bold"))
        self.lbl_result.pack(anchor="w", pady=(0, 20))
        
        self.lbl_time = tk.Label(left_frame, text="Execution time: 00:00", font=("Arial", 9))
        self.lbl_time.pack(anchor="w", side=tk.BOTTOM)

        # ================= KANAN: PREVIEW GAMBAR [cite: 95-98] =================
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(right_frame, text="Face Recognition", font=("Arial", 18, "bold")).pack(pady=(0, 20))
        
        img_container = tk.Frame(right_frame)
        img_container.pack(fill=tk.BOTH, expand=True)
        
        # Kotak Test Image
        box1 = tk.Frame(img_container)
        box1.pack(side=tk.LEFT, expand=True)
        tk.Label(box1, text="Test Image").pack()
        self.canvas_test = tk.Label(box1, bg="gray", width=20, height=10)
        self.canvas_test.pack(pady=10)
        
        # Kotak Closest Result
        box2 = tk.Frame(img_container)
        box2.pack(side=tk.RIGHT, expand=True)
        tk.Label(box2, text="Closest Result").pack()
        self.canvas_result = tk.Label(box2, bg="gray", width=20, height=10)
        self.canvas_result.pack(pady=10)

    def load_dataset(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dataset_folder = folder_selected
            self.lbl_dataset_status.config(text=os.path.basename(folder_selected))
            messagebox.showinfo("Info", "Dataset dipilih. Silakan pilih Test Image untuk memulai proses (Program akan memuat dataset saat eksekusi).")

    def load_test_image(self):
        if not self.dataset_folder:
            messagebox.showwarning("Peringatan", "Harap pilih folder dataset terlebih dahulu!")
            return
            
        file_selected = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_selected:
            self.test_image_path = file_selected
            self.lbl_image_status.config(text=os.path.basename(file_selected))
            
            # Tampilkan gambar test di GUI
            img = Image.open(self.test_image_path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_test.config(image=img_tk, width=150, height=150)
            self.canvas_test.image = img_tk
            
            # Jalankan Proses Pengenalan Wajah
            self.run_recognition()

    def run_recognition(self):
        start_time = time.time()
        
        # 1. BACA DATASET (Feature Extraction dengan PIL) [cite: 64, 65]
        X_train = []
        self.y_train = []
        self.X_train_images = []
        
        for filename in os.listdir(self.dataset_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Format Kaggle PINS biasanya butuh disesuaikan, kita ambil nama file/folder sebagai label
                label = filename.split('_')[0] 
                img_path = os.path.join(self.dataset_folder, filename)
                
                img = Image.open(img_path).convert('L').resize(self.IMG_SIZE)
                img_array = np.array(img).flatten()
                
                X_train.append(img_array)
                self.y_train.append(label)
                self.X_train_images.append(img_path)
                
        X_train = np.array(X_train)
        
        if len(X_train) == 0:
            messagebox.showerror("Error", "Dataset kosong atau format gambar tidak valid.")
            return

        # 2. PROSES ALJABAR LINIER (Eigenface)
        # Rataan matriks (Mean Face) [cite: 42]
        self.mean_face = np.mean(X_train, axis=0)
        X_centered = X_train - self.mean_face # [cite: 43]
        
        # Hitung matriks kovarians (menggunakan A^T * A untuk efisiensi ruang) [cite: 45]
        covariance_matrix = np.dot(X_centered, X_centered.T) 
        
        # Panggil perhitungan Eigen Manual kita [cite: 116, 117]
        k_components = min(20, len(X_train)) # Ambil 20 eigenface terbaik
        eigenvalues, eigenvectors_small = calculate_manual_eigen(covariance_matrix, k_components)
        
        # Kembalikan dimensi vektor eigen ke bentuk piksel gambar
        eigenvectors = np.dot(X_centered.T, eigenvectors_small)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        self.eigenfaces = eigenvectors
        
        # Proyeksi bobot training
        self.weights_train = np.dot(X_centered, self.eigenfaces)

        # 3. PENGUJIAN GAMBAR
        test_img = Image.open(self.test_image_path).convert('L').resize(self.IMG_SIZE)
        test_array = np.array(test_img).flatten()
        test_centered = test_array - self.mean_face
        weight_test = np.dot(test_centered, self.eigenfaces) # [cite: 51]
        
        # 4. CARI KEMIRIPAN DENGAN JARAK EUCLIDEAN MANUAL [cite: 54, 116]
        min_dist = float('inf')
        best_match_idx = -1
        
        for i in range(len(self.weights_train)):
            dist = manual_euclidean_distance(self.weights_train[i], weight_test)
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i
                
        # 5. TAMPILKAN HASIL
        end_time = time.time()
        exec_time = end_time - start_time
        
        predicted_label = self.y_train[best_match_idx]
        best_img_path = self.X_train_images[best_match_idx]
        
        self.lbl_result.config(text=f"Match: {predicted_label}")
        self.lbl_time.config(text=f"Execution time: {exec_time:.2f}s")
        
        # Tampilkan gambar di GUI
        res_img = Image.open(best_img_path).resize((150, 150))
        res_img_tk = ImageTk.PhotoImage(res_img)
        self.canvas_result.config(image=res_img_tk, width=150, height=150)
        self.canvas_result.image = res_img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()