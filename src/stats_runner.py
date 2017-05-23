# Main file for getting AI stats
import numpy as np
import ann as ann

# Run straight ANN VS RND
num_runs = 20


print("ANN 3x3:")

p1wins=0
p1loss=0
draw=0

p2wins=0
p2loss=0

for i in range(num_runs):
	test = ann.rndVsNN(3, False)
	test2 = ann.NNVsRnd(3, False)

	if test == 0:
		draw = draw + 1
	if test == 1:
		p1wins = p1wins+1
	if test == -1:
		p1loss = p1loss+1 

	if test2 == 1:
		p2wins = p2wins+1
	if test2 == -1:
		p2loss = p2loss+1 
	# print("Player 1: Wins: %d, draws: %d, loss: %d" % (p1wins, p1loss, draw))
	# print("Player 2: Wins: %d, draws: %d, loss: %d" % (p2wins, p2loss, draw))

print("Player 1: Wins: %d, draws: %d, loss: %d" % (p1wins, p1loss, draw))
print("Player 2: Wins: %d, draws: %d, loss: %d" % (p2wins, p2loss, draw))

print("ANN 5x5:")

p1wins=0
p1loss=0
draw=0

p2wins=0
p2loss=0

for i in range(num_runs):
	test = ann.rndVsNN(5, False)
	test2 = ann.NNVsRnd(5, False)

	if test == 0:
		draw = draw + 1
	if test == 1:
		p1wins = p1wins+1
	if test == -1:
		p1loss = p1loss+1 

	if test2 == 1:
		p2wins = p2wins+1
	if test2 == -1:
		p2loss = p2loss+1 
	# print("Player 1: Wins: %d, draws: %d, loss: %d" % (p1wins, p1loss, draw))
	# print("Player 2: Wins: %d, draws: %d, loss: %d" % (p2wins, p2loss, draw))

print("Player 1: Wins: %d, draws: %d, loss: %d" % (p1wins, p1loss, draw))
print("Player 2: Wins: %d, draws: %d, loss: %d" % (p2wins, p2loss, draw))

