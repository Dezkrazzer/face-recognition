import torch

def manual_euclidean_distance(v1, v2):
    """
    Menghitung Euclidean distance
    """
    diff = v1 - v2
    return torch.sqrt(torch.sum(diff ** 2)).item()

def manual_eig(A, num_components=50, max_iter=100, tol=1e-5):
    """
    Mencari eigen value & vector dengan Power Iteration + Deflation.
    """
    n = A.shape[0]
    num_components = min(num_components, n) 
    
    device = A.device
    dtype = A.dtype
    
    eigenvalues = torch.zeros(num_components, dtype=dtype, device=device)
    eigenvectors = torch.zeros((n, num_components), dtype=dtype, device=device)
    
    A_k = A.clone()
    
    for i in range(num_components):
        v = torch.rand(n, 1, dtype=dtype, device=device)
        v = v / torch.norm(v)
        
        for _ in range(max_iter):
            v_new = torch.matmul(A_k, v)
            v_new = v_new / torch.norm(v_new)
            
            if torch.norm(v_new - v) < tol or torch.norm(v_new + v) < tol:
                v = v_new
                break
            v = v_new
        
        eig_val = torch.matmul(v.T, torch.matmul(A_k, v))[0, 0]
        
        eigenvalues[i] = eig_val
        eigenvectors[:, i] = v.squeeze()
        
        A_k = A_k - eig_val * torch.matmul(v, v.T)
        
    return eigenvalues, eigenvectors