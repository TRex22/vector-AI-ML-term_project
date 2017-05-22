import numpy as np
import matplotlib.pyplot as pl

def plot_data(x, name=''):
    """
    plots the input data set
    """
    pl.figure()
    pl.scatter(x[0,:], x[1,:],  c='b', marker='o', s=64, label='Data')
    pl.xlabel(r'$x_{1}$', fontsize=16)
    pl.ylabel(r'$x_{2}$', fontsize=16)
    pl.axes().set_aspect('equal','datalim')
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)
    
def compute_mean(x):
    """
    computes and returns the mean vector of the input data
    """
    m = np.mean(x, axis=1)[:, np.newaxis]
    return m
    
def compute_covariance(x):
    """
    computes and returns the covariance matrix of the input data x
    """
    X = x.copy()
    m = compute_mean(X)
    N = X.shape[1]
    X = X - m
    S = 1.0/(N-1) * np.dot(X, X.T)
    return S

def compute_principal_components(x):
    """
    computes and returns the principal components
    """
    # compute the covariance matrix
    S = compute_covariance(x)
    # compute eigenvalues and eigenvectors
    l, p = np.linalg.eigh(S)
    # sort the eigenvectors in descending order
    li = np.argsort(l)[::-1]
    L = np.diag(l[li])
    P = p[:,li]
    return P, L

def select_principal_components(P, L, delta):
    """75
    selects and returns principal components considering
    desired retained variation
    """
    l = np.diagonal(L)
    d = np.argmin(np.abs(delta - np.cumsum(l) / np.sum(l))) + 1
    Pd = P[:,0:d]
    Ld = L[0:d, 0:d]
    return Pd, Ld, d

def plot_principal_components(x, Pd, Ld, name=''):
    """
    plots the principal vectors
    """
    plot_data(x)
    m = compute_mean(x)
    d = Ld.shape[0]
    pl.scatter(m[0,0], m[1,0], c='g', marker='o', s=32)
    colors = ['r', 'g']
    title = ''
    for i in np.arange(d):
        # project the data along the principal component
        w = np.dot((x-m).T,Pd[:, i][:, np.newaxis])
        p = m + np.max(np.abs(w)) * Pd[:, i][:, np.newaxis]
        mp = np.hstack((m,p))
        pl.plot(mp[0,:], mp[1,:], c=colors[i], linewidth=2)
        p = m - np.max(np.abs(w)) * Pd[:, i][:, np.newaxis]
        mp = np.hstack((m,p))
        pl.plot(mp[0,:], mp[1,:], c=colors[i], linewidth=2,label=('PC'+str(i+1)))
        title = title + r'$\lambda_' + str(i+1) + r'$ = ' + '{:.4f}'.format(Ld[i,i]) + ' '
    pl.legend()
    pl.title(title)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)

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