import utils as u
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import h5py


batch_size = 128
num_classes = 2
epochs = 40

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# xinput = np.loadtxt(open("training.txt"), delimiter=",")
# b = a[a[:, 2] > 50.0]
xinput = u.generateGameDataUsingRnd(3, 100000)
np.savez_compressed("million_alphatoe.dat", xinput=xinput) 

xwin = xinput[xinput[:, 20] == 1];
print(xwin.shape)
x_train = xwin[:50000, :20]
y_train = xwin[:50000, 19:21]
reward_train = xwin[:50000, -1]

x_test = xwin[50000:100000, :20]
y_test = xwin[50000:100000, 19:21]
reward_test = xwin[50000:100000, -1]

# x_train = xinput[:50000, :784] #input and out[put]
# y_train = xinput[:50000, -1]
# x_test = xinput[50000 : 60000, :784]
# y_test = xinput[50000 : 60000, -1]

print('x_train.shape: %s \ny_train.shape: %s \nx_test.shape: %s \ny_test.shape: %s' %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(9*9, activation='sigmoid', input_shape=(9,)))
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

outfile = "exp1.dat"

data1 = np.vstack((bias[0], kernel[0]))
data2 = np.vstack((bias[1], kernel[1]))
data3 = np.vstack((bias[2], kernel[2]))
# data4 = np.vstack((bias[3], kernel[3]))

np.savez_compressed(outfile, score=score, data1=data1, data2=data2) #, data3=data3
