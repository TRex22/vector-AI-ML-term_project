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
epochs = 24

board_size = 3
nn = board_size*board_size
num_random_matches = 1000000
half_matches = num_random_matches/2

# the data, shuffled and split between train and test sets
xinput = np.load('data/NN_natural_3_3.dat.npz')
xinput = xinput['xinput']

half_matches = xinput.shape[0]/2

print(xinput.shape)
x_train = xinput[:half_matches, :board_size*board_size]
y_train = xinput[:half_matches, board_size*board_size:2*board_size*board_size]
reward_train = xinput[:half_matches, -1]

x_test = xinput[half_matches:num_random_matches, :board_size*board_size]
y_test = xinput[half_matches:num_random_matches, board_size*board_size:2*board_size*board_size]
reward_test = xinput[half_matches:num_random_matches, -1]

print('x_train.shape: %s \ny_train.shape: %s \nx_test.shape: %s \ny_test.shape: %s' %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(nn, activation='sigmoid', input_shape=(nn,)))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(nn, activation='sigmoid'))

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

data1 = np.vstack((bias[0][0], kernel[0][0]))
data2 = np.vstack((bias[1][0], kernel[1][0]))
data3 = np.vstack((bias[2][0], kernel[2][0]))
data4 = np.vstack((bias[3][0], kernel[3][0]))
data5 = np.vstack((bias[4][0], kernel[4][0]))
data6 = np.vstack((bias[5][0], kernel[5][0]))
# data4 = np.vstack((bias[3], kernel[3]))

np.savez_compressed("data/autoencoder_3.dat", score=score, data1=data1, data2=data2, data3=data3, data4=data4, data5=data5, data6=data6)
