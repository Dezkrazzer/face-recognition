# main.py
import io
import os
import cv2
import torch
from src.utils.eigen import manual_euclidean_distance
from src.screen.main_screen import FaceRecognitionApp

IMG_SIZE = (50, 50) 
UI_IMAGE_SIZE = (150, 150)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# CASCADE STACKING (FRONTAL + PROFILE)
# ==========================================
CASCADE_FRONTAL = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CASCADE_PROFILE = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade_frontal = cv2.CascadeClassifier(CASCADE_FRONTAL)
face_cascade_profile = cv2.CascadeClassifier(CASCADE_PROFILE)

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

    def load_model(self):
        """Membaca dan merakit kembali seluruh pecahan file biner dari folder src/dataset/"""
        model_dir = os.path.join(CURRENT_DIR, "src", "dataset")
        files = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("Eigen_Weight_")])
    
        if not files:
            raise Exception("Tidak ada file pecahan model (Eigen_Weight_*.pt) ditemukan di src/dataset/!")
        
        print(f"[AI] Menemukan {len(files)} pecahan model. Memulai penggabungan biner di RAM...")
        
        full_data = b""
        for f in files:
            with open(f, 'rb') as file:
                full_data += file.read()
            
        buffer = io.BytesIO(full_data)
        checkpoint = torch.load(buffer, map_location=DEVICE, weights_only=False)
    
        self.mean_face = checkpoint['mean_face'].to(DEVICE)
        self.eigenfaces = checkpoint['eigenfaces'].to(DEVICE)
        self.projected_training_faces = checkpoint['projected_training_faces'].to(DEVICE)
        self.dataset_labels = checkpoint['dataset_labels']
        self.dataset_images = checkpoint['dataset_images']
        self.is_loaded = True
        print(f"[AI] Model berhasil disatukan dan dimuat ke {DEVICE.type.upper()}!")

    def recognize(self, test_image_path, threshold):
        img = cv2.imread(test_image_path)
        if img is None: return None, None, float('inf'), "Gambar gagal dibaca."
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # LOGIKA CASCADE STACKING INFERENCE
        faces = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                gray_flipped = cv2.flip(gray, 1)
                faces_flipped = face_cascade_profile.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces_flipped) > 0:
                    faces = []
                    for (x, y, w, h) in faces_flipped:
                        faces.append((gray.shape[1] - x - w, y, w, h))
        
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