import numpy as np
import matplotlib.pyplot as pl
import tic_tac_toe as game

def flattenWorld(world):
	return np.reshape(world, (1, world.shape[0] * world.shape[1]))[0] # col major

def rebuildWorld(world):
	size = world.shape[0]
	return world.reshape(size,size); # should be right orientation

def thresholdOutput(output):
	size = output.shape[0]

	for i in range(size):
		if output[i] >= 0.5:
			output[i] = 1
		elif output[i] < 0.5:
			output[i] = 0

	return output

def rndVsRnd(size):
	world = game.initGameWorld(size)

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
			world = game.rndMove(world, 1.0)
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

def runRndGame(board_size, num_game):
	world_list = np.zeros((board_size*board_size, board_size*board_size+3))

	world = game.initGameWorld(board_size)
	movesLeft = game.numberMovesLeft(world)
	hasWon = False
	moveCount = 0

	while(movesLeft > 0) and (hasWon == False):
		# player 1
		if (movesLeft > 0) and (hasWon == False):
			world, x, y = game.rndMoveXY(world, 1.0)
			hasWon = game.checkWin(world, 1.0)

			if hasWon:
				world_list[moveCount] = np.concatenate((flattenWorld(world), [x], [y], [1]), axis=0)
			elif not hasWon:
				world_list[moveCount] = np.concatenate((flattenWorld(world), [x], [y], [0]), axis=0)

			moveCount = moveCount+1		

		# player 2
		if (movesLeft > 0) and (hasWon == False):
			world, x, y = game.rndMoveXY(world, -1)
			hasWon = game.checkWin(world, -1)

			if hasWon:
				world_list[moveCount] = np.concatenate((flattenWorld(world), [x], [y], [0]), axis=0)
			elif not hasWon:
				world_list[moveCount] = np.concatenate((flattenWorld(world), [x], [y], [1]), axis=0) # world x,y 1

			moveCount = moveCount+1			

		if (movesLeft > 0):
			movesLeft = game.numberMovesLeft(world)

	if(game.checkDraw(world, moveCount)):
		for i in range(world_list.shape[0]):
			world_list[i][board_size*board_size+2] = 0.5

	return world_list


def generateGameDataUsingRnd(board_size, num_game):
	# 3 sets of data
	# perfect move, 1st player no centre and 2nd player
	# xinput -> flattened board | 2 y outputs | outcome/reward
	nn = board_size*board_size
	xinput = np.zeros((num_game*nn, nn+3))

	for i in range(num_game):
		rndGame = runRndGame(board_size, num_game)
		for j in range(rndGame.shape[0]):
			xinput[i+j] = rndGame[j]
		i = i + rndGame.shape[0]-1

	return xinput

xinput = generateGameDataUsingRnd(3, 10)
print(xinput)