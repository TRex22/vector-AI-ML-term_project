# import sys
# sys.path.insert(0, '../utils')
import utils as u
import numpy as np
import h5py


batch_size = 128
num_classes = 2
epochs = 40

num_random_matches = 1000

print("generate data....")
print("3x3")
board_size = 3
xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]

xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+3] == 2] # player 2 draw as player 1 should never draw
xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+2] == 0.5] # 0 is a loss to player 1
print(xinput[xinput[:, 2*board_size*board_size+3] == 2])
print(xwin_player1.shape)
print(xdraw_player2.shape)

# np.savez_compressed("3_3_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

# print("4x4")
# board_size = 4
# xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

# xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]

# xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+3] == 2] # player 2 draw as player 1 should never draw
# xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+3] == 0.5] # 0 is a loss to player 1

# print(xwin_player1.shape)
# print(xdraw_player2.shape)

# np.savez_compressed("4_4_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

# print("5x5")
# board_size = 5
# xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

# xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]

# xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+3] == 2] # player 2 draw as player 1 should never draw
# xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+3] == 0.5] # 0 is a loss to player 1

# print(xwin_player1.shape)
# print(xdraw_player2.shape)

# np.savez_compressed("5_5_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

# num_random_matches = 1000000
# print("10x10")
# board_size = 10
# xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

# xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]

# xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+3] == 2] # player 2 draw as player 1 should never draw
# xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+3] == 0.5] # 0 is a loss to player 1

# print(xwin_player1.shape)
# print(xdraw_player2.shape)

# np.savez_compressed("15_15_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2)