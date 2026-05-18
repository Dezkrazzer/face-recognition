# core_math.py
import numpy as np

class EigenfaceRecognizer:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.weights_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        self.y_train = y_train
        
        # 1. Hitung Wajah Rata-rata & Mean Centering
        self.mean_face = np.mean(X_train, axis=0)
        X_centered = X_train - self.mean_face
        
        # 2. Matriks Kovarians (menggunakan trik A^T * A untuk efisiensi memori)
        covariance_matrix = np.dot(X_centered, X_centered.T)
        
        # 3. Hitung Eigenvalues dan Eigenvectors
        eigenvalues, eigenvectors_small = np.linalg.eig(covariance_matrix)
        
        # 4. Proyeksikan kembali vektor eigen ke dimensi asli
        eigenvectors = np.dot(X_centered.T, eigenvectors_small)
        
        # Normalisasi vektor eigen
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        
        # 5. Urutkan berdasarkan nilai eigen terbesar
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenfaces = eigenvectors[:, sorted_indices][:, :self.n_components]
        
        # 6. Proyeksikan data latih ke "Ruang Wajah" sebagai referensi
        self.weights_train = np.dot(X_centered, self.eigenfaces)

    def predict(self, test_image):
        # Pusatkan data uji
        test_centered = test_image - self.mean_face
        
        # Proyeksikan ke ruang wajah
        weight_test = np.dot(test_centered, self.eigenfaces)
        
        # Cari jarak Euclidean terdekat
        distances = np.linalg.norm(self.weights_train - weight_test, axis=1)
        best_match_index = np.argmin(distances)
        
        return self.y_train[best_match_index], distances[best_match_index], best_match_index