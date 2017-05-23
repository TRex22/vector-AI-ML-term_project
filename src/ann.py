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

	data_dat = np.load('data/NN_natural_3_3.dat.npz')
	
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

	x,y = determine_xy(layer3, size)
	print(x,y)
	
	world[x][y] = player
	return world

def rndVsNN3():
	world = game.initGameWorld(3)

	print("Welcome to TicTacToe!")
	print("\n####################")

	game.printWorld(world)

	movesLeft = game.numberMovesLeft(world)
	hasWon = False
	moveCount = 0

	while(movesLeft > 0) and (hasWon == False):
		# player 1
		if (movesLeft > 0) and (hasWon == False):
			print("Player 1:")
			world = NN3(world, 1.0)
			hasWon = game.checkWin(world, 1.0)

			if hasWon:
				print("Player 1 Won!")
				game.printWorld(world)
			moveCount = moveCount+1		

		# player 2
		if (movesLeft > 0) and (hasWon == False):
			
			print("Player 2:")
			world = game.rndMove(world, -1)
			hasWon = game.checkWin(world, -1)

			if hasWon:
				print("Player 2 Won!")
				game.printWorld(world)
			moveCount = moveCount+1			

		if (movesLeft > 0):
			movesLeft = game.numberMovesLeft(world)

		print("Moves Left: %d\n"%(movesLeft))

	if(game.checkDraw(world, moveCount)):
		print("It's a draw!")

rndVsNN3()