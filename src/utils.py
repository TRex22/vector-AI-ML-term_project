import numpy as np
import matplotlib.pyplot as pl

def standardizeInput(world, player):
	size = world.shape[0]
	return world.flatten('C');

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