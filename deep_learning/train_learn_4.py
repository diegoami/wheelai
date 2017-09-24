# Files can be downlooaded here : https://github.com/hello2all/GTSRB_Keras_STN/blob/master/input/download_data.md
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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')


from collections import Counter
train_label_counter = Counter(y_train)

train_counter = Counter(y_train)
order = list(zip(*train_counter.most_common()))[0]

f, ax = plt.subplots(figsize=(12, 4))
ax = sns.countplot(x=y_train, order=order, color='lightblue', ax=ax, label="train")

_ = ax.set_title('Class distribution')
_ = ax.legend(ncol=2, loc="upper right", frameon=True)

Y_train = np_utils.to_categorical(y_train, 43)
Y_valid = np_utils.to_categorical(y_valid, 43)
Y_test = np_utils.to_categorical(y_test, 43)

print (Y_train.shape)

nb_train_samples = X_train.shape[0]
nb_valid_samples = X_valid.shape[0]
batch_size = 32
steps_per_epoch = nb_train_samples // batch_size
valid_steps = nb_valid_samples // batch_size

def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization()(x)
def relu_bn(x): return relu(bn(x))

def conv(x, nf, sz, wd, p):
    x = Conv2D(nf, (sz, sz), kernel_initializer="he_uniform", padding='same',
                          kernel_regularizer=regularizers.l2(wd))(x)
    return dropout(x,p)

def conv_block(x, nf, bottleneck=False, p=None, wd=0):
    x = relu_bn(x)
    if bottleneck: x = relu_bn(conv(x, nf * 4, 1, wd, p))
    return conv(x, nf, 3, wd, p)


def dense_block(x, nb_layers, growth_rate, bottleneck=False, p=None, wd=0):
    if bottleneck: nb_layers //= 2
    for i in range(nb_layers):
        b = conv_block(x, growth_rate, bottleneck=bottleneck, p=p, wd=wd)
        x = merge([x,b], mode='concat', concat_axis=-1)
    return x


def transition_block(x, compression=1.0, p=None, wd=0):
    nf = int(x.get_shape().as_list()[-1] * compression)
    x = relu_bn(x)
    x = conv(x, nf, 1, wd, p)
    return AveragePooling2D((2, 2), strides=(2, 2))(x)


def create_dense_net(nb_classes, img_input, depth=40, nb_block=3,
                     growth_rate=12, nb_filter=16, bottleneck=False, compression=1.0, p=None, wd=0,
                     activation='softmax'):
    assert activation == 'softmax' or activation == 'sigmoid'
    assert (depth - 4) % nb_block == 0
    nb_layers_per_block = int((depth - 4) / nb_block)
    nb_layers = [nb_layers_per_block] * nb_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    for i, block in enumerate(nb_layers):
        x = dense_block(x, block, growth_rate, bottleneck=bottleneck, p=p, wd=wd)
        if i != len(nb_layers) - 1:
            x = transition_block(x, compression=compression, p=p, wd=wd)

    x = relu_bn(x)
    x = GlobalAveragePooling2D()(x)
    return Dense(nb_classes, activation=activation, kernel_regularizer=regularizers.l2(wd))(x)


input_shape = (32,32,3)
img_input = Input(shape=input_shape)

x = create_dense_net(43, img_input, depth=100, nb_filter=16, compression=0.5,
                     bottleneck=True, p=0.2, wd=1e-4)

model = Model(img_input, x)

model.compile(loss='sparse_categorical_crossentropy',
      optimizer=keras.optimizers.SGD(0.1, 0.9, nesterov=True), metrics=["accuracy"])

K.set_value(model.optimizer.lr, 0.1)

# checkpoint
filepath=results+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, y_train, 64, 20, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list)



model.save(results+'first_model.hp5')
