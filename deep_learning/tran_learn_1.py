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

batch_size = 32
epochs = 3

def get_batches(path, class_mode='categorical', gen=image.ImageDataGenerator(), \
                shuffle=True, target_size=(224,224), batch_size=1):
    return gen.flow_from_directory(path, class_mode=class_mode, batch_size=batch_size, \
                                   target_size=target_size, shuffle=shuffle)

def get_steps(batches, batch_size):
    steps = int(batches.samples/batch_size)
    return (steps if batches.samples%batch_size==0 else (steps+1))


train_b = get_batches(traindata_path, batch_size=batch_size)
valid_b = get_batches(validdata_path, batch_size=batch_size)

train_labels = train_b.classes
valid_labels = valid_b.classes


y_train = keras.utils.to_categorical(train_labels)
y_valid = keras.utils.to_categorical(valid_labels)

print(train_labels.shape, valid_labels.shape, y_train.shape, y_valid.shape)


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


trn_b = get_batches(traindata_path, class_mode=None, shuffle=False, batch_size=batch_size)
val_b = get_batches(validdata_path, class_mode=None, shuffle=False, batch_size=batch_size)
tst_b = get_batches(testdata_path, class_mode=None, shuffle=False, batch_size=batch_size)

trn_steps = get_steps(train_b, batch_size=batch_size)
val_steps = get_steps(valid_b, batch_size=batch_size)
tst_steps = get_steps(tst_b, batch_size=batch_size)

vgg_conv = vgg_conv()
vgg_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trn_ft = vgg_conv.predict_generator(trn_b, trn_steps)
val_ft = vgg_conv.predict_generator(val_b, val_steps)
#tst_ft = vgg_conv.predict_generator(tst_b, tst_steps)

print(trn_ft.shape, val_ft.shape)

save_array(results_path + 'trn_ft.dat', trn_ft)
save_array(results_path + 'val_ft.dat', val_ft)