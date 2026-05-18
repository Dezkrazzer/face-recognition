import numpy as np

# Menguji apakah PyTorch bisa digunakan dan memiliki akses GPU
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        device = torch.device("cuda")
        print("🚀 CUDA GPU Terdeteksi! Menjalankan komputasi dengan PyTorch.")
    else:
        GPU_AVAILABLE = False
        print("⚠️ PyTorch terinstal tetapi CUDA tidak tersedia. Menggunakan mode CPU (NumPy).")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ PyTorch belum terinstal. Menggunakan mode CPU (NumPy).")

def manual_euclidean_distance(vec1, vec2):
    """Menghitung jarak Euclidean secara manual (Wajib sesuai Juknis)"""
    distance = 0.0
    for i in range(len(vec1)):
        distance += (vec1[i] - vec2[i]) ** 2
    return distance ** 0.5

def power_iteration(A, num_iterations=100):
    """Mencari 1 Nilai & Vektor Eigen dominan secara manual"""
    if GPU_AVAILABLE and isinstance(A, torch.Tensor):
        # Jalur Eksekusi GPU (PyTorch)
        n = A.shape[0]
        b_k = torch.ones(n, dtype=A.dtype, device=A.device)
        for _ in range(num_iterations):
            b_k1 = torch.matmul(A, b_k)
            norm = torch.norm(b_k1)
            b_k = b_k1 / norm
        eigenvalue = torch.dot(b_k, torch.matmul(A, b_k)) / torch.dot(b_k, b_k)
        return eigenvalue, b_k
    else:
        # Jalur Eksekusi CPU (NumPy)
        n = A.shape[0]
        b_k = np.ones(n)
        for _ in range(num_iterations):
            b_k1 = np.dot(A, b_k)
            norm = np.linalg.norm(b_k1)
            b_k = b_k1 / norm
        eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
        return eigenvalue, b_k

def calculate_manual_eigen(C, k_components):
    """Mencari k Nilai & Vektor Eigen menggunakan Power Iteration & Deflation"""
    if GPU_AVAILABLE:
        A_gpu = torch.tensor(C, dtype=torch.float64, device=device)
        eigenvalues = []
        eigenvectors = []
        for i in range(k_components):
            print(f"Mencari komponen Eigen ke-{i+1} dari {k_components} (GPU)...")
            val, vec = power_iteration(A_gpu)
            eigenvalues.append(val.item())
            eigenvectors.append(vec.cpu().numpy())
            
            vec_col = vec.reshape(-1, 1)
            A_gpu = A_gpu - val * torch.matmul(vec_col, vec_col.T)
        return np.array(eigenvalues), np.column_stack(eigenvectors)
    else:
        A = C.copy()
        eigenvalues = []
        eigenvectors = []
        for i in range(k_components):
            print(f"Mencari komponen Eigen ke-{i+1} dari {k_components} (CPU)...")
            val, vec = power_iteration(A)
            eigenvalues.append(val)
            eigenvectors.append(vec)
            
            vec_col = vec.reshape(-1, 1)
            A = A - val * np.dot(vec_col, vec_col.T)
        return np.array(eigenvalues), np.column_stack(eigenvectors)