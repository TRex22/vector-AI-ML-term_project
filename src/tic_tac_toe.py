# Jason Chalom 711985 May 2017
# Handler for Tic Tac Toe

import numpy as np
# from random import randInt

def initGameWorld(size):
	world = np.zeros((size, size))

	for i in range(size):
		for j in range(size):
			world[i][j] = 0
	return world

def printWorld(world):
	size = world.shape[0]
	output = "\n\n"

	for k in range(24):
		output += " ="
	output += "\n"

	for i in range(size):
		output += "|\t"
		for j in range(size):
			output += "%d\t|\t" % (world[i][j])
		output += "\n"

	for k in range(24): # size*size*size-size
		output += " ="

	print (output)

def checkMove(world, x, y):
	if world[x][y] == 0.0:
		return True
	
	return False

def numberMovesLeft(world):
	size = world.shape[0]
	count = 0

	for i in range(size):
		for j in range(size):
			if world[i][j] == 0.0:
				count = count + 1.0

	return count

def checkDraw(world, moveCount):
	size = world.shape[0]
	# draw
	if(moveCount == (size**2 - 1.0)):
		return True
	return False

def checkWin(world, player=1.0):
	size = world.shape[0]
	
	rowcount = 0
	colcount = 0
	diag1count = 0
	diag2count = 0

	for i in range(size):
		rowcount = 0
		colcount = 0

		for j in range(size):
			if world[i][j] == player:
				rowcount = rowcount + 1
			if world[j][i] == player:
				colcount = colcount + 1

		if world[i][i] == player:
			diag1count = diag1count + 1

		if world[i][size-1-i] == player:
			diag2count = diag2count + 1

		if rowcount >= size:
			return True
		if colcount >= size:
			return True
		if diag1count == size or diag2count == size:
			return True

		rowcount = 0
		colcount = 0

	return False

def rndMove(world, player):
	size = world.shape[0]

	x = 0.0
	y = 0.0

	madeMove = False

	# can create unending loop if no moves left
	movesLeft = numberMovesLeft(world)
	while madeMove == False:
		rndNumbers = np.random.randint(size, size=2)
		x = rndNumbers[0] # 0 to size-1 ie the coords
		y = rndNumbers[1] # 0 to size-1 ie the coords

		madeMove = checkMove(world, x, y)

	world[x][y] = player
	return world

def rndMoveXY(world, player):
	size = world.shape[0]

	x = 0.0
	y = 0.0

	madeMove = False

	# can create unending loop if no moves left
	movesLeft = numberMovesLeft(world)
	while madeMove == False:
		rndNumbers = np.random.randint(size, size=2)
		x = rndNumbers[0] # 0 to size-1 ie the coords
		y = rndNumbers[1] # 0 to size-1 ie the coords

		madeMove = checkMove(world, x, y)

	world[x][y] = player
	return world, x, y