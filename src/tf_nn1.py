import utils as u
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import h5py


batch_size = 128
# num_classes = 2
epochs = 40

board_size = 3
nn = board_size*board_size
num_random_matches = 1000000
half_matches = num_random_matches/2

# the data, shuffled and split between train and test sets
# xinput = u.generateGameDataUsingRnd(3, num_random_matches)
# np.savez_compressed("million_alphatoe.dat", xinput=xinput) 

xinput = u.generateGameDataUsingRnd(board_size, num_random_matches)

xwin_player1 = xinput[xinput[:, 2*board_size*board_size+3] == 1]
xdraw_player2 = xinput[xinput[:, 2*board_size*board_size+3] == 2] # player 2 draw as player 1 should never draw
xdraw_player2 = xdraw_player2[xdraw_player2[:, 2*board_size*board_size+3] == 0.5] # 0 is a loss to player 1

print(xwin_player1.shape)
x_train = xwin_player1[:half_matches, :board_size*board_size]
y_train = xwin_player1[:half_matches, board_size*board_size:2*board_size*board_size]
reward_train = xwin_player1[:half_matches, -1]

x_test = xwin_player1[half_matches:num_random_matches, :board_size*board_size]
y_test = xwin_player1[half_matches:num_random_matches, board_size*board_size:2*board_size*board_size]
reward_test = xwin_player1[half_matches:num_random_matches, -1]

print('x_train.shape: %s \ny_train.shape: %s \nx_test.shape: %s \ny_test.shape: %s' %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
np.savez_compressed("data/3_3_million.dat", xinput=xinput, xwin_player1=xwin_player1, xdraw_player2=xdraw_player2) 

# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(nn, activation='sigmoid', input_shape=(nn,)))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))
# model.add(Dense(9, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

bias = []
kernel = []

for layer in model.layers:
    weights = layer.get_weights()
    
    kernel.append([weights[0]])
    bias.append([weights[1]])

# print(np.array(bias).shape)                                                                                                                     
# print(np.array(kernel).shape)

print(np.array(bias[0][0]).shape)                                                                                                                     
print(np.array(kernel[0][0]).shape)

data1 = np.concatenate((np.array(bias[0][0]), np.array(kernel[0][0])), axis=0)
data2 = np.concatenate((np.array(bias[1][0]), np.array(kernel[1][0])), axis=0)
data3 = np.concatenate((np.array(bias[2][0]), np.array(kernel[2][0])), axis=0)

# data1 = np.vstack((bias[0], kernel[0]))
# data2 = np.vstack((bias[1], kernel[1]))
# data3 = np.vstack((bias[2], kernel[2]))
# data4 = np.vstack((bias[3], kernel[3]))

np.savez_compressed("data/NN_natural_3_3.dat", score=score, data1=data1, data2=data2, data3=data3) #, data3=data3
