import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm

import utils as util

def im2vec(I):
    """
    converts a 2D image to 1D row vector
    """
    I = I.copy()
    rows, cols = I.shape
    x = np.reshape(I, (1, rows * cols))
    return x

def vec2im(x, rows, cols):
    """
    converts a 1D row vector to 2D image of dimensions rows x cols
    """
    x = x.copy()
    I = np.reshape(x, (rows, cols))
    return I

def normalize_image(I):
    """
    normalizes input image for displaying
    """
    I = I.copy()
    I = I - np.min(I)
    I = I / np.max(I)
    return I

def plot_vector(x, rows, cols, name='', title=''):
    """
    plots a vector x as 2D image4
    """
    I = x.copy()
    I = vec2im(I, rows, cols)
    I = normalize_image(I)
    pl.figure()
    pl.imshow(I, cmap=pl.cm.gray)
    pl.xticks([])
    pl.yticks([])
    pl.title(title, fontsize=16)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)

def plot_eigenvalues(L, name='', title=''):
    pl.figure()
    pl.plot(np.diag(L))
    pl.xlabel(r'$i$', fontsize=16)
    pl.ylabel(r'$\lambda_{i}$', fontsize=16)
    pl.title(title, fontsize=16)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)
    
def read_subject(data_folder, subject_folder):
    """
    reads facial images of a subject
    """
    
    yaw = ['000', '010', '020', '030', '040', '050',
           '060', '070', '080', '090', '100', '110',
           '120', '130', '140', '150', '160', '170', '180']
    pitch = ['060', '070', '080', '090', '100', '110', '120']
    image_path = data_folder + subject_folder + r'/' + subject_folder[:-4] + r'_' + pitch[0] + r'_' + yaw[0] + '.ras'
    I = pl.imread(image_path)
    rows, cols = I.shape
    sf = 2
    rows = rows / sf
    cols = cols / sf
    X = np.zeros(((rows*cols), len(pitch)*len(yaw)))
    for r in np.arange(len(pitch)):
        for c in np.arange(len(yaw)):
            image_path =  data_folder + subject_folder + r'/' + subject_folder[:-4] + r'_' + pitch[r] + r'_' + yaw[c] + '.ras'
            I = pl.imread(image_path)[::sf,::sf] / 255.0
            X[:, (r*len(yaw) + c)] = im2vec(I)[0,:]
    return X, yaw, pitch, rows, cols

def plot_subject(X, rows, cols, yaw, pitch, name):
    """
    plots images of a subject on grid
    """
    for r in np.arange(len(pitch)):
        for c in np.arange(len(yaw)):
            I = vec2im(X[:, (r*len(yaw) + c)], rows, cols)
            if c == 0:
                D = I.copy()
            else:
                D = np.hstack((D,I))
        if r == 0:
            DD = D.copy()
        else:
            DD = np.vstack((DD, D))
    pl.figure()
    pl.imshow(DD, cmap=pl.cm.gray)
    pl.xticks([])
    pl.yticks([])
    pl.xlabel(r'$\leftarrow$ yaw $\rightarrow$', fontsize=16)
    pl.ylabel(r'$\leftarrow$ pitch (tilt) $\rightarrow$', fontsize=16)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)

def plot_subject_at_projections(X, W, rows, cols, yaw, pitch, name):
    pl.figure()
    pl.axes().set_autoscaley_on(False)
    pl.axes().set_autoscalex_on(False)
    pl.axes().set_ylim([-12.0,12.0])
    pl.axes().set_xlim([-12.0,12.0])
    pl.axes().set_aspect('equal','datalim')
    colors = cm.rainbow(np.linspace(0, 1, len(pitch)))
    for r in np.arange(len(pitch)):
        for c in np.arange(len(yaw)):
            i = r*len(yaw) + c
            pl.imshow(vec2im(X[:,i], rows, cols), extent=(W[0,i], W[0,i] + 1, W[1,i], W[1,i] + 1), cmap=pl.cm.gray)
            pl.scatter(W[0,i], W[1,i],  c=colors[r,:], marker='o', s=64)
    pl.xlabel(r'$w_{1}$', fontsize = 16)
    pl.ylabel(r'$w_{2}$', fontsize = 16)
    if name != '':
        pl.savefig(name, bbox_inches='tight', dpi=300)
    

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
   
def createY(K):
    newCY = np.zeros((K,1));
    for i in range(K):
        newCY[i] = i;

    return newCY;

def 


# def flattenArray(arr):
#     rows = arr.shape[0]
#     cols = arr.shape[1]

#     vector = 

#     for i in range(rows):
#         for j in range(cols):

# def saveImage(arr, filename='result'):
#     Y = createY(arr.shape[0])
#     util.arrRecord2im(arr, Y, 0, filename, 18);

# pl.close('all')

# # load data
# input_data = util.load_data();
# X_im = input_data[0];
# Y_im = input_data[1];

# rows = X_im.shape[0]
# cols = X_im.shape[1]

# X = np.array((X_im))#np.reshape(X_im, (rows * cols, 1))
# m = compute_mean(X)
# P, L = compute_principal_components(X)
# delta = 0.95
# Pd, Ld, d = select_principal_components(P, L, delta)
# # print(Pd.shape)
# # print(Ld.shape)
# # print(d.shape)
# # saveImage(Pd)

# for i in np.arange(6):
#     name = 'images/_eigenvector' + str(i+1) + '.png'
#     plot_vector(Pd[:,i], rows, cols, name, title=r'$\lambda_' + str(i+1) + '$ = ' + '{:.2f}'.format(L[i,i]))
    