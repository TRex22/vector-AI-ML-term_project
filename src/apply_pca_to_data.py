import numpy as np
import pca as pca

data_dat = np.load('data/3_3_million.dat.npz')
x = data_dat['xdraw_player2'].T
# print(.shape)

 # compute the mean vector of the data set
m = pca.compute_mean(x)
# compute the covariance matrix of the data set
S = pca.compute_covariance(x)
# compute all principal components
P, L = pca.compute_principal_components(x)
# select principal components according to desired variation
delta = 1.0
Pd, Ld, d = pca.select_principal_components(P, L, delta)

# print(Pd.T)

W = pca.project_onto_eigenvector_space(x, Pd, m)

# reconstruct using d-rank approximation
d_range = [1, 5, 10, 20, 50, 100, 150]
i = 66

for d in d_range:
    Pd = P[:,0:d]
    W =  pca.project_onto_eigenvector_space(x, Pd, m)
    Xhat = pca.reconstruct_data(Pd, W, m)
    # name = '../../slides/docs/' + subject_folder + '_reconstruction' + str(d) + '.png'
    # plot_vector(Xhat[:,i], rows, cols, name, title=r'$d=' + str(d) + '$')
    print(Xhat[:,i])

