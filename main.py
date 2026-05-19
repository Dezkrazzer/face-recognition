import os
import cv2
import numpy as np
import torch
from src.utils.eigen import manual_eig, manual_euclidean_distance
from src.screen.main_screen import FaceRecognitionApp

IMG_SIZE = (50, 50) 
UI_IMAGE_SIZE = (150, 150)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "src", "dataset", "eigen_weights.pt")

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

torch.set_float32_matmul_precision('medium') 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceRecognizer:
    """Kelas murni berisi logika matematis dan AI (Tanpa UI)"""
    def __init__(self):
        self.is_loaded = False
        self.mean_face = None
        self.eigenfaces = None
        self.projected_training_faces = None
        self.dataset_labels = []
        self.dataset_images = [] 

    def build_dataset(self, dataset_path, num_components=50):
        if not os.path.exists(dataset_path):
            raise Exception("Folder dataset tidak ditemukan!")

        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, file))

        images, labels, saved_ui_images = [], [], []
        total_files = len(all_files)
        
        print(f"\n[AI] Mulai kompilasi {total_files} gambar ke Portable Model (.pt)...")
        
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
                continue

            resized_gray = cv2.resize(gray_crop, IMG_SIZE)
            images.append(resized_gray.flatten())
            
            resized_color = cv2.resize(color_crop, UI_IMAGE_SIZE)
            saved_ui_images.append(cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB))
            labels.append(os.path.basename(os.path.dirname(img_path))) 
            
            if (idx + 1) % 500 == 0:
                print(f" -> Progress: {idx + 1} / {total_files} gambar terekstraksi...")
                    
        Gamma_np = np.array(images).T
        if Gamma_np.shape[1] == 0:
            raise Exception("Dataset kosong atau wajah tidak terdeteksi.")

        M = Gamma_np.shape[1]
        print("\n[AI] Menghitung Aljabar Linear di VRAM...")
        Gamma = torch.tensor(Gamma_np, dtype=torch.float32, device=DEVICE)
        mean_face = torch.mean(Gamma, dim=1, keepdim=True)
        Phi = Gamma - mean_face
        C = torch.matmul(Phi.T, Phi) / M
        
        eigenvalues, eigenvectors = manual_eig(C, num_components=num_components)
        
        U = torch.matmul(Phi, eigenvectors)
        U = U / torch.norm(U, dim=0, keepdim=True)
        projected_faces = torch.matmul(U.T, Phi)
        
        checkpoint = {
            'mean_face': mean_face.cpu(),
            'eigenfaces': U.cpu(),
            'projected_training_faces': projected_faces.cpu(),
            'dataset_labels': labels,
            'dataset_images': saved_ui_images 
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"[AI] Model Portable berhasil dibuat! Tersimpan: {MODEL_PATH}")

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"File {MODEL_PATH} tidak ditemukan.")
            
        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        self.mean_face = checkpoint['mean_face'].to(DEVICE)
        self.eigenfaces = checkpoint['eigenfaces'].to(DEVICE)
        self.projected_training_faces = checkpoint['projected_training_faces'].to(DEVICE)
        self.dataset_labels = checkpoint['dataset_labels']
        self.dataset_images = checkpoint['dataset_images']
        self.is_loaded = True
        print(f"[AI] Model Portable di-load ke VRAM ({DEVICE.type.upper()}). Siap!")

    def recognize(self, test_image_path, threshold):
        img = cv2.imread(test_image_path)
        if img is None: return None, None, float('inf'), "Gambar gagal dibaca."
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            gray_cropped = gray[y:y+h, x:x+w]
        else:
            return None, None, float('inf'), "Wajah tidak terdeteksi oleh Haar Cascade."
        
        resized = cv2.resize(gray_cropped, IMG_SIZE)
        flattened = resized.flatten().reshape(-1, 1)
        
        test_tensor = torch.tensor(flattened, dtype=torch.float32, device=DEVICE)
        Phi_test = test_tensor - self.mean_face
        projected_test_face = torch.matmul(self.eigenfaces.T, Phi_test)
        
        min_dist = float('inf')
        min_idx = -1
        
        for i in range(self.projected_training_faces.shape[1]):
            train_vector = self.projected_training_faces[:, i].view(-1, 1)
            dist = manual_euclidean_distance(projected_test_face, train_vector)
            
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        if min_dist < threshold:
            closest_img = self.dataset_images[min_idx]
            label = self.dataset_labels[min_idx]
            return closest_img, label, min_dist, "Cocok"
        else:
            return None, None, min_dist, "Tidak Dikenal"


if __name__ == "__main__":

    recognizer_logic = FaceRecognizer()
    

    app = FaceRecognitionApp(recognizer=recognizer_logic)
 
    app.mainloop()