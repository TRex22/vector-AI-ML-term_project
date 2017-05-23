# import tic_tac_toe as ttt
# ttt.rndVsRnd(3)
# data_dat['xwin_player1']

# import numpy as np
# data_dat = np.load('data/3_3_million.dat.npz')

# print(data_dat['xdraw_player2'].shape)

import utils as u
import numpy as np

# board_size = 3
# nn = board_size*board_size
# num_random_matches = 1000


# xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

# xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]
# xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+2] == 2] # player 2 draw as player 1 should never draw
# xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+3] == 0.5] # 0 is a loss to player 1

# half_matches = xwin_player1.shape[0]/2

# print(xwin_player1.shape)
# print(xdraw_player2.shape)

# x_train = xwin_player1[:half_matches, :board_size*board_size]
# y_train = xwin_player1[:half_matches, board_size*board_size:2*board_size*board_size]
# reward_train = xwin_player1[:half_matches, -1]

# x_test = xwin_player1[half_matches:num_random_matches, :board_size*board_size]
# y_test = xwin_player1[half_matches:num_random_matches, board_size*board_size:2*board_size*board_size]
# reward_test = xwin_player1[half_matches:num_random_matches, -1]

# print('x_train.shape: %s \ny_train.shape: %s \nx_test.shape: %s \ny_test.shape: %s' %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# # np.savez_compressed("data/3_3_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

# # x_train /= 255
# # x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

data_dat = np.load('data/3_3_million.dat.npz')
xinput = data_dat['xinput']

print(xinput[0])