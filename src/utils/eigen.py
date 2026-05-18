# src/core_math.py
import numpy as np

def manual_euclidean_distance(vec1, vec2):
    """Menghitung jarak Euclidean secara manual """
    distance = 0.0
    for i in range(len(vec1)):
        distance += (vec1[i] - vec2[i]) ** 2
    return distance ** 0.5

def power_iteration(A, num_iterations=100):
    """Mencari 1 Nilai Eigen dan Vektor Eigen dominan secara manual """
    n = A.shape[0]
    # Inisialisasi vektor acak
    b_k = np.ones(n) 
    
    for _ in range(num_iterations):
        # Hitung A * b_k
        b_k1 = np.dot(A, b_k)
        
        # Hitung norm secara manual untuk normalisasi
        norm = sum(x**2 for x in b_k1) ** 0.5
        b_k = b_k1 / norm

    # Hitung Nilai Eigen (Rayleigh Quotient): (b_k^T * A * b_k) / (b_k^T * b_k)
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    return eigenvalue, b_k

def calculate_manual_eigen(C, k_components):
    """Mencari k Nilai & Vektor Eigen menggunakan Power Iteration & Hotelling's Deflation [cite: 116, 117]"""
    A = C.copy()
    eigenvalues = []
    eigenvectors = []
    
    for i in range(k_components):
        # 1. Cari eigen dominan
        val, vec = power_iteration(A)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        
        # 2. Deflation: Kurangi matriks A dengan komponen eigen yang sudah ditemukan
        # Rumus: A_baru = A_lama - lambda * (v * v^T)
        vec = vec.reshape(-1, 1)
        A = A - val * np.dot(vec, vec.T)
        
    return np.array(eigenvalues), np.column_stack(eigenvectors)