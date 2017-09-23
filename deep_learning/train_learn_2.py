from __future__ import division, print_function


DATA_DIR = '/media/diego/QData/datasets/kaggle_cats/'
#DATA_DIR = '/media/shreyas/DATA/ML_DATA/dogsvscats/sample/'
traindata_path = DATA_DIR + 'train/'
validdata_path = DATA_DIR + 'validation/'
testdata_path = DATA_DIR + 'test/'
results_path = DATA_DIR + 'results/'

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications import VGG16

from deep_learning.utils import *
from deep_learning.vgg16 import Vgg16


X_train = load_array(results_path + 'trn_ft.dat')
X_valid = load_array(results_path + 'val_ft.dat')

print(X_train.shape)
print(X_valid.shape)

def vgg_conv():
    conv_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    return conv_model


def top_layer(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))

    return model


batch_size = 32
epochs = 3
train_b = get_batches(traindata_path, batch_size=batch_size)
valid_b = get_batches(validdata_path, batch_size=batch_size)

train_labels = train_b.classes
valid_labels = valid_b.classes
y_train = keras.utils.to_categorical(train_labels)
y_valid = keras.utils.to_categorical(valid_labels)


model = top_layer(X_train.shape[1:])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(X_valid, y_valid))

model.save_weights(results_path+'bottleneck.h5')