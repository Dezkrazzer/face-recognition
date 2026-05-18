# src/main.py
import os
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
from utils.eigen import calculate_manual_eigen, manual_euclidean_distance

# ==========================================
# KONFIGURASI TEMA UI MODERN
# ==========================================
ctk.set_appearance_mode("System")  # Mengikuti tema Windows (Dark/Light)
ctk.set_default_color_theme("blue") # Aksen warna biru ala Tailwind

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition - FATISDA UNS")
        self.geometry("900x550")
        
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
        # Konfigurasi Grid Utama
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= KIRI: SIDEBAR KONTROL =================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1) # Spacer

        ctk.CTkLabel(self.sidebar_frame, text="⚙️ Control Panel", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 30))

        # Dataset Section
        ctk.CTkLabel(self.sidebar_frame, text="Insert Your Dataset", font=ctk.CTkFont(size=14, weight="bold")).grid(row=1, column=0, padx=20, pady=(0, 5), sticky="w")
        self.btn_dataset = ctk.CTkButton(self.sidebar_frame, text="Choose Folder", command=self.load_dataset, fg_color="#3b82f6", hover_color="#2563eb")
        self.btn_dataset.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="ew")
        self.lbl_dataset_status = ctk.CTkLabel(self.sidebar_frame, text="No Folder Chosen", text_color="gray")
        self.lbl_dataset_status.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # Image Section
        ctk.CTkLabel(self.sidebar_frame, text="Insert Your Image", font=ctk.CTkFont(size=14, weight="bold")).grid(row=4, column=0, padx=20, pady=(0, 5), sticky="w")
        self.btn_image = ctk.CTkButton(self.sidebar_frame, text="Choose File", command=self.load_test_image, fg_color="#3b82f6", hover_color="#2563eb")
        self.btn_image.grid(row=5, column=0, padx=20, pady=(0, 5), sticky="ew")
        self.lbl_image_status = ctk.CTkLabel(self.sidebar_frame, text="No File Chosen", text_color="gray")
        self.lbl_image_status.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="nw")

        # Result Text Section (Kiri Bawah)
        ctk.CTkLabel(self.sidebar_frame, text="Result:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=7, column=0, padx=20, pady=(0, 5), sticky="w")
        self.lbl_result = ctk.CTkLabel(self.sidebar_frame, text="None", font=ctk.CTkFont(size=16, weight="bold"), text_color="gray")
        self.lbl_result.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="w")
        
        self.lbl_time = ctk.CTkLabel(self.sidebar_frame, text="Execution time: 00:00", font=ctk.CTkFont(size=12))
        self.lbl_time.grid(row=9, column=0, padx=20, pady=(0, 20), sticky="w")

        # ================= KANAN: PREVIEW GAMBAR =================
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.main_frame, text="Face Recognition", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, columnspan=2, pady=(10, 30))

        # Kotak Test Image
        self.box_test = ctk.CTkFrame(self.main_frame)
        self.box_test.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.box_test, text="Test Image", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.canvas_test = ctk.CTkLabel(self.box_test, text="(Image Preview)", width=200, height=200, fg_color=("gray80", "gray20"), corner_radius=10)
        self.canvas_test.pack(pady=20, expand=True)

        # Kotak Closest Result
        self.box_result = ctk.CTkFrame(self.main_frame)
        self.box_result.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.box_result, text="Closest Result", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        self.canvas_result = ctk.CTkLabel(self.box_result, text="(Match Preview)", width=200, height=200, fg_color=("gray80", "gray20"), corner_radius=10)
        self.canvas_result.pack(pady=20, expand=True)

    def load_dataset(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dataset_folder = folder_selected
            self.lbl_dataset_status.configure(text=os.path.basename(folder_selected))

    def load_test_image(self):
        if not self.dataset_folder:
            # Menggunakan messagebox standar karena ctk belum punya bawaan
            messagebox.showwarning("Peringatan", "Harap pilih folder dataset terlebih dahulu!")
            return

        file_selected = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_selected:
            self.test_image_path = file_selected
            self.lbl_image_status.configure(text=os.path.basename(file_selected))

            # Render image using CTkImage for high DPI support
            img = Image.open(self.test_image_path)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
            self.canvas_test.configure(image=ctk_img, text="")
            
            self.run_recognition()

    def run_recognition(self):
        start_time = time.time()
        
        self.lbl_result.configure(text="Processing...", text_color="#3b82f6")
        self.update()

        X_train = []
        self.y_train = []
        self.X_train_images = []
        
        MAX_IMAGES = 150
        loaded_count = 0

        for root_dir, dirs, files in os.walk(self.dataset_folder):
            for filename in files:
                if loaded_count >= MAX_IMAGES:
                    break
                    
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = os.path.basename(root_dir)
                    if label.startswith("pins_"):
                        label = label.replace("pins_", "")

                    img_path = os.path.join(root_dir, filename)

                    try:
                        img = Image.open(img_path).convert('L').resize(self.IMG_SIZE)
                        img_array = np.array(img).flatten()
                        X_train.append(img_array)
                        self.y_train.append(label)
                        self.X_train_images.append(img_path)
                        loaded_count += 1
                    except Exception as e:
                        print(f"Gagal memuat {img_path}: {e}")
            
            if loaded_count >= MAX_IMAGES:
                break

        X_train = np.array(X_train)

        if len(X_train) == 0:
            self.lbl_result.configure(text="Error: Data Kosong", text_color="#ef4444")
            return

        # Proses Eigenface
        self.mean_face = np.mean(X_train, axis=0)
        X_centered = X_train - self.mean_face 
        covariance_matrix = np.dot(X_centered, X_centered.T)

        k_components = min(50, len(X_train)) 
        eigenvalues, eigenvectors_small = calculate_manual_eigen(covariance_matrix, k_components)

        eigenvectors = np.dot(X_centered.T, eigenvectors_small)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        self.eigenfaces = eigenvectors
        self.weights_train = np.dot(X_centered, self.eigenfaces)

        # Uji Gambar
        test_img = Image.open(self.test_image_path).convert('L').resize(self.IMG_SIZE)
        test_array = np.array(test_img).flatten()
        test_centered = test_array - self.mean_face
        weight_test = np.dot(test_centered, self.eigenfaces) 

        # Cari Kemiripan
        min_dist = float('inf')
        best_match_idx = -1

        for i in range(len(self.weights_train)):
            dist = manual_euclidean_distance(self.weights_train[i], weight_test)
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i

        THRESHOLD = 4500.0  
        print(f"Jarak kemiripan terdekat: {min_dist:.2f}")

        end_time = time.time()
        exec_time = end_time - start_time
        self.lbl_time.configure(text=f"Execution time: {exec_time:.2f}s")

        if min_dist <= THRESHOLD:
            predicted_label = self.y_train[best_match_idx]
            best_img_path = self.X_train_images[best_match_idx]

            self.lbl_result.configure(text=f"Match:\n{predicted_label}", text_color="#10b981") # Warna Emerald/Green

            res_img = Image.open(best_img_path)
            ctk_res_img = ctk.CTkImage(light_image=res_img, dark_image=res_img, size=(200, 200))
            self.canvas_result.configure(image=ctk_res_img, text="")
        else:
            self.lbl_result.configure(text="No Result", text_color="#ef4444") # Warna Red
            self.canvas_result.configure(image="", text="(No Match Found)")

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()