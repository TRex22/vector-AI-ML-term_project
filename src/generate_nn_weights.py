import ann_train as annt
import utils as u
import numpy as np
import h5py

batch_size = 128
# num_classes = 2
epochs = 24

board_size = 3
nn = board_size*board_size
num_random_matches = 1000000
half_matches = num_random_matches/2

# the data, shuffled and split between train and test sets
xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

xwin_player1 = xinput[xinput[:, -1] == 1]
xdraw_player2 = xinput[xinput[:, -2] == 2] # player 2 draw as player 1 should never draw
xdraw_player2 = xdraw_player2[xdraw_player2[:, -1] == 0.5] # 0 is a loss to player 1

half_matches = xwin_player1.shape[0]/2
# xwin_player1 = np.hstack((xwin_player1, xdraw_player2))

print(xwin_player1.shape)
x_train = xwin_player1[:800000, :board_size*board_size]
y_train = xwin_player1[:800000, board_size*board_size:2*board_size*board_size]
reward_train = xwin_player1[:800000, -1]

x_test = xwin_player1[200000:1000000, :board_size*board_size]
y_test = xwin_player1[200000:1000000, board_size*board_size:2*board_size*board_size]
reward_test = xwin_player1[200000:1000000, -1]

print('x_train.shape: %s \ny_train.shape: %s \nx_test.shape: %s \ny_test.shape: %s' %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
np.savez_compressed("data/3_3_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



np.random.seed(7)
theta = dict()
theta[0] = np.random.randn(x_train.shape[1], x_train.shape[1]/2) * 0.01
theta[1] = np.random.randn(x_train.shape[1]/2, y_train.shape[1]) * 0.01
alpha = 0.1
# train ffnn
theta = ffnn_learn(x_train, y_train, alpha, theta)

# after training test on testing data
n = y_test.shape[0] #777
# x = x_testing[n, :][np.newaxis,:]
# y = y_testing[n, :][np.newaxis,:]

error = 0.0

for i in range(n):
	h = ffnn_classify(x, theta)
	

