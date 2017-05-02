'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
#from __future__ import print_function
import cPickle
import gzip
import numpy as np
import theano #novo
import matplotlib.pyplot as plt
np.random.seed(500)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import visualize

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 52, 52
# number of convolutional filters to use
nb_filters = 72
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cPickle.load(gzip.open('base2.pkl.gz', 'rb'))

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)

#visualize.plot_conv_weights(model.layers_['Convolution2D'])

print('Test score:', score[0])
print('Test accuracy:', score[1])

listPredict = []

listPredict = model.predict_classes(X_test)

vp, fp, vn, fn = 0, 0, 0, 0

for i in xrange(X_test.shape[0]):
    if y_test[i] == 0 and listPredict[i] == 0:
        vn += 1
    elif y_test[i] == 0 and listPredict[i] == 1:
        fp += 1
    elif y_test[i] == 1 and listPredict[i] == 1:
        vp += 1
    else:
        fn += 1

sen = float((vp * 1.0) / (vp + fn))

esp = float((vn * 1.0) / (vn + fp))

acu = float(((vp + vn) * 1.0)/ (vn + vp + fp + fn))
##print (sen, esp)

print ("\nSensitivity: " + str(sen) + " Specificity: " + str(esp) + " Acuracy: " + str(acu))

##print (sen, esp)

print ("\n\n")

print ("vp: " + str(vp) + " vn: " + str(vn) + " fn: " + str(fn) + " fp: " + str(fp))
