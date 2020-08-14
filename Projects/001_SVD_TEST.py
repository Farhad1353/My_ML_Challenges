import numpy as np

def reorder_eig_vectors(eig_tuple):
    eig_values, eig_vectors = eig_tuple
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    return eig_values, eig_vectors

# Computing the augmented matrix
def preU_matrix(A):
    return np.matmul(A, A.T)

def U_matrix(A):
    AA_T = preU_matrix(A)
    U_ = np.linalg.eig(AA_T)
    U_eigvalues, U_eigvectors = reorder_eig_vectors(U_)
    U,_ = np.linalg.qr(U_eigvectors)
    return U_eigvalues, U

# Computing the augmented matrix
def preV_matrix(A):
    return np.matmul(A.T, A)

# Computing V matrix
def V_matrix(A):
    A_TA = preV_matrix(A)
    V_ = np.linalg.eig(A_TA)
    V_eigvalues, V_eigvectors = reorder_eig_vectors(V_)
    V,_ = np.linalg.qr(V_eigvectors)
    return V_eigvalues, V


# Constructing our Sigma matrix
def S_matrix(A):
    rows, columns = np.shape(A)
    U_ = U_matrix(A)
    
    # Getting only appropriate number of eigenvalues in correct order
    eigvalues,_ = reorder_eig_vectors(U_)
    min_dim = min(rows, columns)
    eigvalues = eigvalues[0:min_dim]
    
    S = np.zeros((rows,columns))
    np.fill_diagonal(S,np.sqrt(eigvalues))
    return S


def svd(A):
    U,_ = U_matrix(A)
    S = S_matrix(A)
    V,_ = V_matrix(A)
    return U, S, V.T


from utils import rgb2gray

# Uses SVD to construct matrix approximations
def reduced_svd(A, R):
    U, S, V_T = np.linalg.svd(A)
    
    # Obtaining approximation of matrices with R components
    U_hat = U[:,0:R]
    S_hat = S[0:R]
    V_T_hat = V_T[0:R,:]
    return U_hat, S_hat, V_T_hat

# Computes apprimation of original matrix
def approx(A, R):
    U_hat, S_hat, V_T_hat = reduced_svd(A, R)
    S_hat_matrix = np.zeros((S_hat.shape[0], S_hat.shape[0]))
    np.fill_diagonal(S_hat_matrix, S_hat)

    return np.matmul(U_hat, np.matmul(S_hat_matrix, V_T_hat))

import matplotlib.pyplot as plt

# Importing our image
img = plt.imread("images/2017-audi-a7.jpg")
img = rgb2gray(img)

print("The original image has a Matrix size of {} X {} with total entities of {:,d} \n".format(img.shape[0],img.shape[1],img.size))

# Determining parameters
R_vals = [5, 10, 50, 100, 500]

# Displaying approximation
for R in R_vals:
    new_approx = approx(img, R)
    print("R =", R)
    plt.figure()
    plt.imshow(new_approx, cmap='gray')
    plt.show()