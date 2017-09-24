root_data = "/home/diego/qdata/datasets/traffic_signs/"
train_data = root_data + "train.p"
valid_data = root_data + "valid.p"
test_data = root_data + "test.p"
results = root_data + "results/"


import sys
sys.path.append('..')
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
from keras.layers.merge import Concatenate
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.backend import tf as k

import matplotlib.pyplot as plt
import seaborn as sns

from deep_learning.utils import *
import importlib
import deep_learning.utils2;
from deep_learning.utils2 import *

# Load pickled data
import pickle

with open(train_data, mode='rb') as f:
    train = pickle.load(f)
with open(test_data, mode='rb') as f:
    test = pickle.load(f)
with open(valid_data, mode='rb') as f:
    valid = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

limit_mem()

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 43)
Y_valid = np_utils.to_categorical(y_valid, 43)
Y_test = np_utils.to_categorical(y_test, 43)


model = keras.models.load_model(results+'first_model.hp5')
result = model.predict(X_train)
score, acc = model.evaluate(X_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)


