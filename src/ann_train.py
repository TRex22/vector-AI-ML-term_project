import numpy as np
import matplotlib.pyplot as pl    

def load_data():
    """
    loads training and testing data
    """
    
    cls = ['3', '4', '6'] # 3: Happy, 4: Sad, 6: Neutral
    cls_names = ['Happy', 'Sad', 'Neutral']
    # training data    
    loc = '../data/train/'
    train_K = 500
    test_K = 500
    x_training = np.zeros((train_K * len(cls), 24 * 24), dtype=np.float)
    y_training = np.zeros((train_K * len(cls), len(cls)), dtype=np.float)
    k = 0
    for n in np.arange(train_K):
        for c in np.arange(len(cls)):
            # load image
            I = pl.imread(loc + cls[c] + '_' + str(n) + '.jpg')[:,:,0]
            x_training[k,:] = np.reshape(I, (I.shape[0] * I.shape[1]))
            y_training[k,c] = 1.0
            k = k + 1
    # add all one column to the first axis
    x_training = np.hstack((np.ones((x_training.shape[0],1)), x_training))
    
    # testing data
    loc = '../data/test/'
    x_testing = np.zeros((test_K * len(cls), 24 * 24), dtype=np.float)
    y_testing  = np.zeros((test_K * len(cls), len(cls)), dtype=np.float)
    k = 0
    for n in np.arange(test_K):
        for c in np.arange(len(cls)):
            # load image
            I = pl.imread(loc + cls[c] + '_' + str(n) + '.jpg')[:,:,0]
            x_testing[k,:] = np.reshape(I, (I.shape[0] * I.shape[1]))
            y_testing[k,c] = 1.0
            k = k + 1
    # add all one column to the first axis
    x_testing = np.hstack((np.ones((x_testing.shape[0],1)), x_testing))
    return x_training, y_training, x_testing, y_testing

def normalization_parameters(x):
    """
    computes and returns normalization parameters (mean and std vectors of each 
    column) from data x where it is assumed that the very first column is all 
    1s
    """
    mean_vector = np.mean(x, axis=0)
    mean_vector[0] = 0
    std_vector = np.std(x, axis=0)
    std_vector[0] = 1.0
    return mean_vector, std_vector

def normalize_data(x, mean_vector, std_vector):
    """
    normalizes the input data x by parameters mean_vector and std_vector
    """
    x = x - mean_vector
    x = x / std_vector
    return x
    
def act(s):
    """
    activation function
    """
    u = 1.0 / (1.0 + np.exp(-s))
    return u
    
def is_converged(theta, theta_p, epsilon=1e-8):
    """
    checks convergence
    theta: theta at iteration t
    thetap: theta at iteration t-1
    epsilon: a small number used for thresholdin
    """
    L = len(theta)
    d = np.max(np.abs(theta[0]-theta_p[0]))
    for l in np.arange(L):
        d_temp = np.max(np.abs(theta[l]-theta_p[l]))
        if d_temp > d:
            d = d_temp
    if  d <= epsilon:
        return True
    else:
        return False
    
def ffnn_learn(x, y, alpha, theta):
    """
    learns and returns the parameters of ffnn using SGD
    """
    N = x.shape[0]
    L = len(theta)
    converged = False
    epoch = 0
    while not converged:
        mse = 0.0
        for n in np.arange(N): # for each observation
            xn = x[n, :][np.newaxis, :]
            yn = y[n, :][np.newaxis, :]
            s = dict()
            u = dict()
            s[0] = xn.copy()
            u[0] = s[0].copy()
            for l in np.arange(1, L+1): # apply forward propagation
                s[l] = np.dot(u[l-1], theta[l-1])
                u[l] = act(s[l])
                if l != L:
                    u[l][0,0] = 1.0                
            en = u[l] - yn # error of the network
            mse += np.sum(en**2)
            theta_p = theta.copy()
            for l in np.arange(L-1, -1, -1): # back propagation
                delta = (u[l+1] * (1.0 - u[l+1]) * en)
                theta[l] = theta[l] - alpha * np.dot(u[l].T, delta)
                if l != L-1:
                    theta[l][0,0] = 0.0                    
                # accumulate the error at the previous layer
                en = np.dot(en, theta[l].T)
            converged = is_converged(theta, theta_p)
            if converged:
                break;
        epoch += 1
        mse = mse / n
        msg = 'epoch: {:7d}, MSE: {:.6f}'.format(epoch, mse)
        print(msg)
    return theta

def ffnn_classify(x, theta):
    """
    classifies x according ffnn architecture
    """
    L = len(theta)
    s = dict()
    u = dict()
    s[0] = x.copy()
    u[0] = s[0].copy()
    for l in np.arange(1, L+1): # apply forward propagation
        s[l] = np.dot(u[l-1], theta[l-1])
        u[l] = act(s[l])
        if l != L:
            u[l][0,0] = 1.0
    return u[l]
    


