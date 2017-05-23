# Jason Chalom 711985 May 2017
# Artificial Neural Network helper functions for vector AI

import numpy as np
import utils as u
import tic_tac_toe as game

def determine_xy(output_layer, b_s):
	size = output_layer.shape[0]
	# print(output_layer)
	max = -1;
	idex = -1;
	for i in range(size):
		if(output_layer[i] > max):
			max = output_layer[i];
			idex = i;
	if (idex == -1):
		return -1;
	x = idex/b_s
	y = idex%b_s # idex%size-1
	return x,y;

def auto_encode_3(world):
	data_dat = np.load('data/autoencoder_3.dat.npz')
	
	data1 = np.vstack((data_dat['data1']))
	x = np.hstack(([1.0], u.flattenWorld(world)))
	layer1 = np.dot(x, data1)
	layer1 = 1.0/(1.0 + np.exp(-layer1))

	data2 = np.vstack((data_dat['data2']))
	layer1 = np.hstack(([1.0], layer1))
	layer2 = np.dot(layer1, data2)
	layer2 = 1.0/(1.0 + np.exp(-layer2))

	data3 = np.vstack((data_dat['data3']))
	layer2 =  np.hstack(([1.0], layer2))
	layer3 = np.dot(layer2, data3)
	layer3 = 1.0/(1.0 + np.exp(-layer3))

	data4 = np.vstack((data_dat['data4']))
	layer3 =  np.hstack(([1.0], layer3))
	layer4 = np.dot(layer3, data4)
	layer4 = 1.0/(1.0 + np.exp(-layer4))

	return layer4;

def auto_decode_3(layer4):
	data_dat = np.load('data/autoencoder_3.dat.npz')
	data5 = np.vstack((data_dat['data5']))
	layer4 =  np.hstack(([1.0], layer4))
	layer5 = np.dot(layer4, data5)
	layer5 = 1.0/(1.0 + np.exp(-layer5))

	data6 = np.vstack((data_dat['data6']))
	layer5 =  np.hstack(([1.0], layer5))
	layer6 = np.dot(layer5, data6)
	layer6 = 1.0/(1.0 + np.exp(-layer6))

	return layer6

def auto_encode_5(world):
	data_dat = np.load('data/autoencoder_5.dat.npz')
	
	data1 = np.vstack((data_dat['data1']))
	x = np.hstack(([1.0], u.flattenWorld(world)))
	layer1 = np.dot(x, data1)
	layer1 = 1.0/(1.0 + np.exp(-layer1))

	data2 = np.vstack((data_dat['data2']))
	layer1 = np.hstack(([1.0], layer1))
	layer2 = np.dot(layer1, data2)
	layer2 = 1.0/(1.0 + np.exp(-layer2))

	data3 = np.vstack((data_dat['data3']))
	layer2 =  np.hstack(([1.0], layer2))
	layer3 = np.dot(layer2, data3)
	layer3 = 1.0/(1.0 + np.exp(-layer3))

	data4 = np.vstack((data_dat['data4']))
	layer3 =  np.hstack(([1.0], layer3))
	layer4 = np.dot(layer3, data4)
	layer4 = 1.0/(1.0 + np.exp(-layer4))

	return layer4;

def auto_decode_5(layer4):
	data_dat = np.load('data/autoencoder_5.dat.npz')
	data5 = np.vstack((data_dat['data5']))
	layer4 =  np.hstack(([1.0], layer4))
	layer5 = np.dot(layer4, data5)
	layer5 = 1.0/(1.0 + np.exp(-layer5))

	data6 = np.vstack((data_dat['data6']))
	layer5 =  np.hstack(([1.0], layer5))
	layer6 = np.dot(layer5, data6)
	layer6 = 1.0/(1.0 + np.exp(-layer6))

	return layer6

def NN3(world, player):
	size = world.shape[0]
	movesLeft = game.numberMovesLeft(world)
	if movesLeft == 0:
		return world

	madeMove = False
	data_dat = np.load('data/NN_natural_3_3.dat.npz')
	tempWorld = world.copy()
	iterations = 0

	while madeMove == False and iterations < 1000:
		data1 = np.vstack((data_dat['data1']))
		x = np.hstack(([1.0], u.flattenWorld(tempWorld)))
		layer1 = np.dot(x, data1)
		layer1 = 1.0/(1.0 + np.exp(-layer1))

		data2 = np.vstack((data_dat['data2']))
		layer1 = np.hstack(([1.0], layer1))
		layer2 = np.dot(layer1, data2)
		layer2 = 1.0/(1.0 + np.exp(-layer2))

		data3 = np.vstack((data_dat['data3']))
		layer2 =  np.hstack(([1.0], layer2))
		layer3 = np.dot(layer2, data3)
		layer3 = 1.0/(1.0 + np.exp(-layer3))

		x,y = determine_xy(layer3, size)
		# tempWorld[x][y] = player
		madeMove = game.checkMove(tempWorld, x, y)
		# tempWorld[x][y] = 0
		iterations = iterations+1

	if(iterations >= 1000):
		world, x, y = game.rndMoveXY(world, 1)
	else:
		world[x][y] = player
	
	return world,x,y

# 'data/NN_natural_5_5.dat.npz'
def NN5(world, player):
	size = world.shape[0]
	movesLeft = game.numberMovesLeft(world)
	if movesLeft == 0:
		return world

	madeMove = False
	data_dat = np.load('data/NN_natural_5_5.dat.npz')
	tempWorld = world.copy()
	iterations = 0

	while madeMove == False and iterations < 1000:
		data1 = np.vstack((data_dat['data1']))
		x = np.hstack(([1.0], u.flattenWorld(tempWorld)))
		layer1 = np.dot(x, data1)
		layer1 = 1.0/(1.0 + np.exp(-layer1))

		data2 = np.vstack((data_dat['data2']))
		layer1 = np.hstack(([1.0], layer1))
		layer2 = np.dot(layer1, data2)
		layer2 = 1.0/(1.0 + np.exp(-layer2))

		data3 = np.vstack((data_dat['data3']))
		layer2 =  np.hstack(([1.0], layer2))
		layer3 = np.dot(layer2, data3)
		layer3 = 1.0/(1.0 + np.exp(-layer3))

		x,y = determine_xy(layer3, size)
		# tempWorld[x][y] = player
		madeMove = game.checkMove(tempWorld, x, y)
		# tempWorld[x][y] = 0
		iterations = iterations+1

	if(iterations >= 1000):
		world, x, y = game.rndMoveXY(world, 1)
	else:
		world[x][y] = player
	
	return world,x,y

def rndVsNN(board_size, tprint=False):
	world = game.initGameWorld(board_size)
	movesLeft = game.numberMovesLeft(world)
	hasWon = False
	player1won = False
	player2won = False
	moveCount = 0

	while(movesLeft > 0) and (hasWon == False):
		# player 1
		if (movesLeft > 0) and (hasWon == False):
			newWorld, x, y = game.rndMoveXY(world, -1)
			hasWon = game.checkWin(newWorld, 1) 
			
			if hasWon:
				player1won = True

			moveCount = moveCount+1	
            
		if (movesLeft > 0):
			movesLeft = game.numberMovesLeft(world)
            
		# player 2
		if (movesLeft > 0) and (hasWon == False):
			if board_size == 3:
				newWorld, x, y = NN3(world, 1)
			elif board_size == 5:
				newWorld, x, y = NN5(world, 1)
			else:
				newWorld, x, y = game.rndMoveXY(world, 1)

			hasWon = game.checkWin(newWorld, 1)

			if hasWon and not player1won:
				player2won = True
			# print(newWorld)
			world = newWorld
			moveCount = moveCount+1			

		if (movesLeft > 0):
			# print(world)
			# game.printWorld(world)
			movesLeft = game.numberMovesLeft(world)
            
	if(tprint):
		if(game.checkDraw(world, moveCount)):
			print("It's a draw!")

		else:
			if player1won:
				print("player 1 wins!")
			else:
				print("player 2 wins!")
		game.printWorld(world)

	if(game.checkDraw(world, moveCount)):
		return 0

	else:
		if player1won:
			return 1
		else:
			return -1

def NNVsRnd(board_size, tprint=False):
	world = game.initGameWorld(board_size)
	movesLeft = game.numberMovesLeft(world)
	hasWon = False
	player1won = False
	player2won = False
	moveCount = 0

	while(movesLeft > 0) and (hasWon == False):
		# player 1
		if (movesLeft > 0) and (hasWon == False):
			if board_size == 3:
				newWorld, x, y = NN3(world, 1)
			elif board_size == 5:
				newWorld, x, y = NN5(world, 1)
			else:
				newWorld, x, y = game.rndMoveXY(world, 1)

			hasWon = game.checkWin(world, 1) 
			
			if hasWon:
				player1won = True

			moveCount = moveCount+1	
            
		if (movesLeft > 0):
			movesLeft = game.numberMovesLeft(world)
            
		# player 2
		if (movesLeft > 0) and (hasWon == False):
			newWorld, x, y = game.rndMoveXY(world, -1)
			hasWon = game.checkWin(world, -1)

			if hasWon and not player1won:
				player2won = True

			world = newWorld
			moveCount = moveCount+1			

		if (movesLeft > 0):
			movesLeft = game.numberMovesLeft(world)
            
	if(tprint):
		if(game.checkDraw(world, moveCount)):
			print("It's a draw!")

		else:
			if player1won:
				print("player 1 wins!")
			else:
				print("player 2 wins!")
		game.printWorld(world)

	if(game.checkDraw(world, moveCount)):
		return 0

	else:
		if player1won:
			return 1
		else:
			return -1