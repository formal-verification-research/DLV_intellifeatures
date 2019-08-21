#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import struct
from array import array as pyarray
from PIL import Image
import json
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.utils.conv_utils import convert_kernel

# for mnist
from keras.datasets import mnist


#

import mnist as mm

import glob
import h5py
import os


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def read_dataset():

    # parameters for neural network
    batch_size = 128
    nb_classes = 10
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
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

    return (X_train,Y_train,X_test,Y_test, batch_size, nb_epoch)

def build_model():
    """
    define neural network model
    """

    model = Sequential()
    
    model.add(Conv2D(trainable=True, filters = nb_filters, use_bias=True, bias_regularizer=None, input_dtype="float32", batch_input_shape=[None,img_rows, img_cols,1], activation="relu", kernel_initializer="glorot_uniform", kernel_constraint=None, activity_regularizer=None, padding="valid", strides=[1, 1], name="convolution2d_1", bias_constraint=None, data_format="channels_last", kernel_regularizer=None, kernel_size=(nb_conv, nb_conv)))
    #model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=(1, img_rows, img_cols)))
    #model.add(Activation('relu'))
    model.add(Conv2D(kernel_initializer="glorot_uniform", kernel_constraint=None, activity_regularizer=None, trainable=True, padding="valid", strides=[1, 1], filters=32, use_bias=True, name="convolution2d_2", bias_regularizer=None, bias_constraint=None, data_format="channels_last", kernel_regularizer=None, activation="relu", kernel_size=(nb_conv, nb_conv)))
    #model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(name="maxpooling2d_1", trainable=True, data_format="channels_last", pool_size=[nb_pool, nb_pool], padding="valid", strides=[2, 2]))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(rate=0.25, trainable=True, name="dropout_1"))
    #model.add(Dropout(rate=0.25, trainable = True,))

    model.add(Flatten())
    model.add(Dense(name="dense_1", bias_regularizer=None, bias_constraint=None, activity_regularizer=None, trainable=True, kernel_constraint=None, kernel_regularizer=None, input_dim=None, units=128, kernel_initializer="glorot_uniform", use_bias=True, activation="relu"))
    #model.add(Dense(units=128))
    #model.add(Activation('relu'))
    model.add(Dropout(rate=0.5, trainable=True, name="dropout_2"))
    #model.add(Dropout(rate=0.5, trainable = True,))
    model.add(Dense(name="dense_2", bias_regularizer=None, bias_constraint=None, activity_regularizer=None, trainable=True, kernel_constraint=None, kernel_regularizer=None, input_dim=None, units=nb_classes, kernel_initializer="glorot_uniform", use_bias=True, activation="softmax"))
    #model.add(Dense(units=nb_classes))
    #model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model



def read_model_from_file(modelFile):
    """
    define neural network model
    :return: network model
    """

    model = build_model()
    model.summary()

    #weights = sio.loadmat(weightFile)
    #model = model_from_json(open(modelFile).read())

    model = load_model(modelFile, compile=False)
    


   # for (idx,lvl) in [(1,0),(2,2),(3,7),(4,10)]:

    #    weight_1 = 2 * idx - 2
     #  weight_2 = 2 * idx - 1

      #  w1 = weights['weights'][0,weight_1]
        #w1 = np.transpose(w1, (3, 2, 1, 0))
       # w2 = weights['weights'][0,weight_2].flatten()
       # print(w1.shape)

       # model.layers[lvl].set_weights([w1, w2])

    return model



"""
   The following function gets the activations for a particular layer
   for an image in the test set.
   FIXME: ideally I would like to be able to
          get activations for a particular layer from the inputs of another layer.
"""

def getImage(model,n_in_tests):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test /= 255

    Y_test = np_utils.to_categorical(y_test, nb_classes)
    image = X_test[n_in_tests:n_in_tests+1]
    return np.squeeze(image)

def readImage(path):

    import cv2

    im = cv2.resize(cv2.imread(path), (img_rows, img_cols)).astype('float32')
    im = im / 255
    #im = im.transpose(2, 0, 1)

    print("ERROR: currently the reading of MNIST images are not correct, so the classifications are incorrect. ")



    return np.squeeze(im)

def getActivationValue(model,layer,image):

    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    activations = get_activations(model, layer, image)
    return np.squeeze(activations)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

def predictWithImage(model,newInput):

    newInput2 = np.expand_dims(np.expand_dims(newInput, axis=0), axis=0)
    predictValue = model.predict(newInput2)
    newClass = np.argmax(np.ravel(predictValue))
    confident = np.amax(np.ravel(predictValue))
    return (newClass,confident)

def getWeightVector(model, layer2Consider):
    weightVector = []
    biasVector = []

    for layer in model.layers:
         index=model.layers.index(layer)
         h=layer.get_weights()
         if len(h) > 0 and index in [0,2]  and index <= layer2Consider:
         # for convolutional layer
             ws = h[0]
             bs = h[1]

             #print("layer =" + str(index))
             #print(layer.input_shape)
             #print(ws.shape)
             #print(bs.shape)

             # number of filters in the previous layer
             m = len(ws)
             # number of features in the previous layer
             # every feature is represented as a matrix
             n = len(ws[0])

             for i in range(1,m+1):
                 biasVector.append((index,i,h[1][i-1]))

             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1):
                     # (feature, filter, matrix)
                     weightVector.append(((index,j),(index,i),v[j-1]))

         elif len(h) > 0 and index in [7,10]  and index <= layer2Consider:
         # for fully-connected layer
             ws = h[0]
             bs = h[1]

             # number of nodes in the previous layer
             m = len(ws)
             # number of nodes in the current layer
             n = len(ws[0])

             for j in range(1,n+1):
                 biasVector.append((index,j,h[1][j-1]))

             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1):
                     weightVector.append(((index-1,i),(index,j),v[j-1]))
         #else: print "\n"

    return (weightVector,biasVector)

def getConfig(model):

    config = model.get_config()
    config = json.loads((json.dumps(config)))
    #print(config)
    config = [ getLayerName(dict) for dict in config['layers'] ]
    config = zip(range(len(config)),config)
    return config

def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation':
        return dict.get('config').get('activation')
    else:
        return className
