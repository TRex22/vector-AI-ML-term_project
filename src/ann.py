# Jason Chalom 711985 May 2017
# Artificial Neural Network helper functions for vector AI

import numpy as np
import utils as u

def determineMove(output_layer):
	x = -1
	y = -1

	output_layer = u.thresholdOutput(output_layer)
	output_layer = u.rebuildWorld(output_layer)

	size = output_layer.shape[0]

	for i in range(size):
		for j in range(size):
			if (output_layer[i][j] == 1):
				return i,j

	return x,y;

# 9x9x9 with bias at each level
def basic_3_NN(world, player):
	x = u.standardizeInput(world, player)



	x,y = determineClass(layer4);
	return x,y;