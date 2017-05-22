import numpy as np
import matplotlib.pyplot as pl

def compute_mean(x):
    """
    computes and returns the mean vector of the input data
    """
    m = np.mean(x, axis=1)[:, np.newaxis]
    return m
    
def compute_covariance(X):
    """
    computes and returns the covariance matrix of the input data x
    """
    X = X.copy()
    m = compute_mean(X)
    N = X.shape[1]
    X = X - m
    S = 1.0/(N-1) * np.dot(X, X.T)
    return S

def compute_principal_components(X):
    """
    computes and returns the principal components
    """
    # compute the covariance matrix
    S = compute_covariance(X)
    # compute eigenvalues and eigenvectors
    l, p = np.linalg.eigh(S)
    # sort the eigenvectors in descending order
    li = np.argsort(l)[::-1]
    L = np.diag(l[li])
    P = p[:,li]
    return P, L

def select_principal_components(P, L, delta):
    """
    selects and returns principal components considering
    desired retained variation
    """
    l = np.diagonal(L)
    d = np.argmin(np.abs(delta - np.cumsum(l) / np.sum(l))) + 1
    Pd = P[:,0:d]
    Ld = L[0:d, 0:d]
    return Pd, Ld, d

def project_onto_eigenvector_space(X, Pd, m):
    """
    projects data matrix onto eigenvector space
    """
    W = np.dot(Pd.T, (X-m))
    return W

def reconstruct_data(Pd, W, m):
    """
    reconstructs data matrix using d-rank approximation
    """
    Xhat = m + np.dot(Pd, W)
    return Xhat    

# pl.close('all')   
# # generate a synthetic data set1
# x = generate_data()
# # compute the mean vector of the data set
# m = compute_mean(x)
# # compute the covariance matrix of the data set
# S = compute_covariance(x)
# # compute all principal components
# P, L = compute_principal_components(x)
# # select principal components according to desired variation
# delta = 1.0
# Pd, Ld, d = select_principal_components(P, L, delta)
# # plot data and principal components
# plot_principal_components(x, Pd, Ld, '../../slides/docs/principal_components4.png')