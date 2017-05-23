import numpy as np
import utils as u
import tic_tac_toe as game

import ann as ann

board_size = 3
nn = board_size*board_size
num_random_matches = 1000000
half_matches = num_random_matches/2

data_dat = np.load('data/3_3_million.dat.npz')
xinput = data_dat['xinput']

# new_xinput = np.zeros((xinput.shape[0], 2*3+4))

for i in range(xinput.shape[0]):
	inp = xinput[i][:board_size*board_size]
	out = xinput[i][board_size*board_size:2*board_size*board_size]

	inp = ann.auto_encode_3(u.rebuildWorld(inp, board_size))
	out = ann.auto_encode_3(u.rebuildWorld(out, board_size))

	xinput[i] = np.concatenate(([inp], [out], xinput[i][-4], xinput[i][-3], [2], [0]), axis=0) 

xwin_player1 = xinput[xinput[:, -1] == 1]
xdraw_player2 = xinput[xinput[:, -2] == 2] # player 2 draw as player 1 should never draw
xdraw_player2 = xdraw_player2[xdraw_player2[:, -1] == 0.5] # 0 is a loss to player 1

np.savez_compressed("data/3_3_million_autoenc.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 
